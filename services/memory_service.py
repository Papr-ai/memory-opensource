from typing import Dict, Any, Optional, List
from fastapi import BackgroundTasks, HTTPException
from models.parse_server import AddMemoryResponse, AddMemoryItem
from memory.memory_graph import MemoryGraph, AsyncSession
from services.user_utils import User
from services.logging_config import get_logger
from services.connector_service import find_user_by_connector_ids
from core.services.telemetry import get_telemetry
import json
from memory.memory_graph import MemoryGraph
from memory.memory_item import (
    TextMemoryItem, CodeSnippetMemoryItem, DocumentMemoryItem,
    WebpageMemoryItem, CodeFileMemoryItem, MeetingMemoryItem,
    PluginMemoryItem, IssueMemoryItem, CustomerMemoryItem
)
from models.parse_server import UpdateMemoryResponse,  AddMemoryResponse, AddMemoryItem, BatchMemoryError, BatchMemoryResponse
from os import environ as env
from dotenv import find_dotenv, load_dotenv
from redis.asyncio import Redis
from services.logging_config import get_logger
from urllib.parse import urlparse
import uuid
from models.memory_models import AddMemoryRequest
import httpx
from models.shared_types import MemoryMetadata

from services.logger_singleton import LoggerSingleton
from services.memory_policy_resolver import (
    resolve_memory_policy_from_schema,
    extract_omo_fields_from_policy,
    should_skip_graph_extraction
)

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

logger = get_logger(__name__)

# Log at module level to verify logger is working
logger.info("Memory routes module loaded")

# Telemetry is initialized via get_telemetry() when needed
# No need to initialize Amplitude client anymore

# Get API key for hotglue
hotglue_api_key = env.get("HOTGLUE_PAPR_API_KEY")
logger.info(f"hotglue_api_key inside memory_routes.py: {hotglue_api_key}")

# Initialize chat_gpt
chat_gpt = None  # This should be properly initialized based on your application's needs


async def batch_handle_incoming_memories(
    memory_requests: List[AddMemoryRequest],
    end_user_id: str,
    developer_user_id: str,
    sessionToken: str,
    neo_session: AsyncSession,
    user_info: Optional[Dict[str, Any]],
    client_type: str,
    memory_graph: MemoryGraph,
    background_tasks: BackgroundTasks,
    skip_background_processing: bool = False,
    user_workspace_ids: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    legacy_route: bool = True,
    workspace_id: Optional[str] = None,
    api_key_id: Optional[str] = None
) -> List[AddMemoryResponse]:
    """
    Batch version of handle_incoming_memory - processes multiple memories in a single transaction.
    
    Args:
        memory_requests: List of AddMemoryRequest objects to process
        ... (same args as handle_incoming_memory)
        
    Returns:
        List of AddMemoryResponse objects
    """
    logger.info(f"📦 batch_handle_incoming_memories: Processing {len(memory_requests)} memories")
    
    try:
        # Check memory limits (use first memory for size calculation, or sum all)
        developer_user = User(developer_user_id)
        logger.info(f"developer_user: {developer_user}")
        
        # Extract organization_id and namespace_id from first memory_request if available
        organization_id = memory_requests[0].organization_id if memory_requests and hasattr(memory_requests[0], 'organization_id') else None
        namespace_id = memory_requests[0].namespace_id if memory_requests and hasattr(memory_requests[0], 'namespace_id') else None
        
        # Calculate total memory size in MB from all contents
        total_memory_size_bytes = sum(len(req.content.encode('utf-8')) for req in memory_requests)
        total_memory_size_mb = total_memory_size_bytes / (1024 * 1024)
        logger.info(f"Total memory size: {total_memory_size_bytes} bytes ({total_memory_size_mb:.4f} MB)")
        
        limit_check = await developer_user.check_memory_limits(
            workspace_id=workspace_id,
            organization_id=organization_id,
            namespace_id=namespace_id,
            api_key_id=api_key_id,
            memory_size_mb=total_memory_size_mb
        )
        if limit_check is not None:
            error_response, status_code = limit_check
            # Return error for all memories
            if isinstance(error_response, dict):
                error_msg = error_response.get("message", error_response.get("error", "Unknown error"))
            else:
                error_msg = str(error_response)
            return [AddMemoryResponse.failure(error=error_msg, code=status_code, details=error_response) 
                   for _ in memory_requests]

        # Track batch memory creation with telemetry
        try:
            telemetry = get_telemetry()
            await telemetry.track(
                "batch_memory_created", 
                {
                    "count": len(memory_requests),
                    "total_size_mb": total_memory_size_mb,
                }, 
                user_id=end_user_id,
                developer_id=developer_user_id
            )
        except Exception as e:
            logger.error(f"Error tracking batch memory creation: {e}")

        # Process all memories
        memory_items = []
        relationships_lists = []
        developer_user_object_id = None
        
        for memory_request in memory_requests:
            # Process each memory_request similar to handle_incoming_memory
            content = memory_request.content
            memory_type = memory_request.type
            metadata = memory_request.metadata.model_dump() if memory_request.metadata else {}
            
            # Resolve external user IDs (only once, use first memory)
            if not developer_user_object_id and memory_request.metadata and getattr(memory_request.metadata, 'external_user_id', None):
                async with httpx.AsyncClient() as client:
                    user_service = User(developer_user_id)
                    updated_metadata, resolved, _ = await user_service.resolve_external_user_ids_to_internal(
                        developer_id=developer_user_id,
                        metadata=memory_request.metadata,
                        httpx_client=client,
                        x_api_key=api_key
                    )
                    memory_request.metadata = updated_metadata
                    developer_user_object_id = resolved.get("developerUser_objectId")
                    logger.info(f"resolved: {resolved}")
            
            # Handle metadata and ACL (simplified for batch - use same ACL for all)
            metadata['user_id'] = str(end_user_id)
            metadata['workspace_id'] = workspace_id if workspace_id is not None else None
            
            # Add createdAt if not present
            if 'createdAt' not in metadata:
                from datetime import datetime, timezone
                metadata['createdAt'] = datetime.now(timezone.utc).isoformat()
            
            # Sanitize metadata
            metadata = MemoryGraph.sanitize_metadata(metadata)
            
            # Create MemoryItem
            context = [c.model_dump() for c in memory_request.context] if memory_request.context else []
            
            if memory_type in ['text', 'TextMemoryItem', 'message']:
                from memory.memory_item import TextMemoryItem
                memory_item = TextMemoryItem(content, metadata, context)
            elif memory_type in ['document', 'DocumentMemoryItem']:
                from memory.memory_item import DocumentMemoryItem
                memory_item = DocumentMemoryItem(content, metadata, context)
            else:
                from memory.memory_item import TextMemoryItem
                memory_item = TextMemoryItem(content, metadata, context)
            
            memory_items.append(memory_item)
            
            # Handle relationships
            relationships_json = [r.model_dump() for r in memory_request.relationships_json] if memory_request.relationships_json else []
            from models.memory_models import RelationshipItem
            relationship_items = [RelationshipItem(**rel) if isinstance(rel, dict) else rel for rel in relationships_json]
            relationships_lists.append(relationship_items)
        
        # Extract graph generation configuration (use first memory's config for all)
        first_request = memory_requests[0]
        graph_override = None
        schema_id = None
        simple_schema_mode = False
        property_overrides = None
        
        if first_request.graph_generation:
            graph_gen = first_request.graph_generation
            if graph_gen.mode == "manual" and graph_gen.manual:
                graph_override = graph_gen.manual
                logger.info(f"🎯 BATCH MANUAL MODE: Using developer-provided graph structure")
            elif graph_gen.mode == "auto" and graph_gen.auto:
                auto_config = graph_gen.auto
                schema_id = auto_config.schema_id
                simple_schema_mode = auto_config.simple_schema_mode
                property_overrides = auto_config.property_overrides
                logger.info(f"🤖 BATCH AUTO MODE: schema_id={schema_id}, simple_schema_mode={simple_schema_mode}")
        
        # Call batch processing method
        try:
            stored_memories = await memory_graph.batch_add_memory_items_async(
                memory_items=memory_items,
                relationships_json_list=relationships_lists,
                sessionToken=sessionToken,
                user_id=end_user_id,
                background_tasks=background_tasks,
                neo_session=neo_session,
                add_to_pinecone=True,
                workspace_id=workspace_id,
                skip_background_processing=skip_background_processing,
                user_workspace_ids=user_workspace_ids,
                api_key=api_key,
                developer_user_object_id=developer_user_object_id,
                legacy_route=legacy_route,
                developer_user_id=developer_user_id,
                graph_override=graph_override,
                schema_id=schema_id,
                simple_schema_mode=simple_schema_mode,
                property_overrides=property_overrides
            )
        except Exception as e:
            logger.error(f"Error adding batch memory items to graph: {str(e)}")
            return [AddMemoryResponse.failure(error="Error adding memory items", code=500, details=str(e))
                   for _ in memory_requests]
        
        if not stored_memories:
            return [AddMemoryResponse.failure(error="No memories stored", code=500)
                   for _ in memory_requests]
        
        # Return individual responses
        responses = []
        for stored_memory in stored_memories:
            responses.append(AddMemoryResponse.success(
                data=[
                    AddMemoryItem(
                        memoryId=stored_memory.memoryId,
                        createdAt=stored_memory.createdAt,
                        objectId=stored_memory.objectId,
                        memoryChunkIds=stored_memory.memoryChunkIds
                    )
                ]
            ))
        
        logger.info(f"✅ batch_handle_incoming_memories: Completed {len(responses)} memories")
        return responses
        
    except Exception as e:
        logger.error(f"❌ Error in batch_handle_incoming_memories: {str(e)}", exc_info=True)
        # Return error for all memories
        return [AddMemoryResponse.failure(error=str(e), code=500) for _ in memory_requests]


async def handle_incoming_memory(
    memory_request: AddMemoryRequest,
    end_user_id: str,
    developer_user_id: str,
    sessionToken: str,
    neo_session: AsyncSession,
    user_info: Optional[Dict[str, Any]],
    client_type: str,
    memory_graph: MemoryGraph,
    background_tasks: BackgroundTasks,
    skip_background_processing: bool = False,
    user_workspace_ids: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    legacy_route: bool = True,
    workspace_id: Optional[str] = None,
    api_key_id: Optional[str] = None
) -> AddMemoryResponse:
    """
    Handle a single memory item addition.
    Returns an AddMemoryResponse.
    """
    try:
        # Check memory limits
        developer_user = User(developer_user_id)
        logger.info(f"developer_user: {developer_user}")
        
        # Calculate memory size in MB from content
        content = memory_request.content if memory_request else ""
        memory_size_bytes = len(content.encode('utf-8'))
        memory_size_mb = memory_size_bytes / (1024 * 1024)  # Convert bytes to MB
        logger.info(f"Memory size: {memory_size_bytes} bytes ({memory_size_mb:.4f} MB)")
        
        # Extract organization_id and namespace_id for limit checking
        # These should have been set by apply_multi_tenant_scoping_to_memory_request() in the route
        organization_id = getattr(memory_request, 'organization_id', None) if memory_request else None
        namespace_id = getattr(memory_request, 'namespace_id', None) if memory_request else None
        
        limit_check = await developer_user.check_memory_limits(
            workspace_id=workspace_id,
            organization_id=organization_id,
            namespace_id=namespace_id,
            api_key_id=api_key_id,
            memory_size_mb=memory_size_mb
        )
        if limit_check is not None:
            error_response, status_code = limit_check
            if isinstance(error_response, dict):
                return AddMemoryResponse.failure(
                    error=error_response.get("message", error_response.get("error", "Unknown error")),
                    code=status_code,
                    details=error_response
                )
            else:
                return AddMemoryResponse.failure(
                    error=str(error_response),
                    code=status_code
                )

        # Track memory creation with telemetry (privacy-first)
        try:
            telemetry = get_telemetry()
            await telemetry.track(
                "memory_created", 
                {
                    "type": memory_request.type if hasattr(memory_request, 'type') else "unknown",
                    "has_metadata": bool(memory_request.metadata) if hasattr(memory_request, 'metadata') else False,
                }, 
                user_id=end_user_id,  # The developer's end user
                developer_id=developer_user_id  # The API key owner / developer
            )
        except Exception as e:
            logger.error(f"Error tracking memory creation: {e}")
            # Continue processing even if telemetry fails
        
        # Use AddMemoryRequest fields directly
        content = memory_request.content
        memory_type = memory_request.type
        logger.info(f"memory_type: {memory_type}")
        metadata = memory_request.metadata.model_dump() if memory_request.metadata else {}
        logger.info(f"Type of metadata handle_add_memory: {type(metadata)}")

        # Add a log before extracting memory_type_metadata
        logger.info("Attempting to extract 'type' from metadata.")
        memory_type_metadata = metadata.get('type')
        logger.info(f"memory_type inside metadata: {memory_type_metadata}")

        is_private = metadata.get('is_private', True)
        logger.info(f"is_private: {is_private}")

        # If memory_type is 'message', adjust user_id and sessionToken
        if memory_type_metadata == 'message':
            connector = metadata.get('connector')
            connector_user_id = metadata.get('user')
            logger.info(f"Connector user ID from metadata: {connector_user_id}")

            if connector and connector_user_id:
                # Retrieve ACL object IDs using the connector_service
                acl_object_ids = await find_user_by_connector_ids(sessionToken, connector, [connector_user_id])
                logger.info(f"ACL Object IDs: {acl_object_ids}")

                if acl_object_ids:
                    real_user_id = acl_object_ids[0]
                    logger.info(f"Real user ID: {real_user_id}")

                    # Lookup session token for the real user
                    real_sessionToken = await User.lookup_user_token(real_user_id)
                    logger.info(f"Real sessionToken: {real_sessionToken}")

                    # Verify the session token
                    real_user_info = await User.verify_session_token(real_sessionToken)
                    logger.info(f"Verified real user info: {real_user_info}")

                    if real_user_info:
                        end_user_id = real_user_id
                        sessionToken = real_sessionToken
                        logger.info("Attributed memory to the actual message creator.")
                    else:
                        logger.error("Session token verification failed for real user.")
                        logger.info("Using original user_id and sessionToken (Slack admin).")
                else:
                    logger.warning("No ACL Object IDs found for connector user.")
                    logger.info("Using original user_id and sessionToken (Slack admin) because message creator doesn't have a Papr account.")
            else:
                logger.error("Missing connector or user ID in metadata.")
                logger.info("Using original user_id and sessionToken (Slack admin) due to missing connector or user ID.")

        additional_user_ids = metadata.get('additional_user_ids', [])
        context = [c.model_dump() for c in memory_request.context] if memory_request.context else []
        relationships_json = [r.model_dump() for r in memory_request.relationships_json] if memory_request.relationships_json else []
        project_id = metadata.get('project_id')
        # Use the workspace_id passed as parameter, fallback to metadata if not provided
        if workspace_id is None:
            workspace_id = metadata.get('workspace_id')
        logger.info(f"workspace_id: {workspace_id}")
        sourceType = metadata.get('sourceType')
        logger.info(f"sourceType: {sourceType}")
        sourceUrl = metadata.get('sourceUrl')
        logger.info(f"sourceUrl: {sourceUrl}")
        postMessageId = metadata.get('postMessageId')
        logger.info(f"postMessageId: {postMessageId}")
        pageId = metadata.get('pageId')
        logger.info(f"pageId: {pageId}")
        url = metadata.get('url')
        logger.info(f"url: {url}")
        connector = metadata.get('connector')
        logger.info(f"connector: {connector}")

        # Ensure metadata is a dict
        if not isinstance(metadata, dict):
            metadata = {}

        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                return AddMemoryResponse.failure(
                    error="Invalid metadata format",
                    code=400
                )

        if not relationships_json:
            relationships_json = []

        if isinstance(relationships_json, str):
            try:
                relationships_json = json.loads(relationships_json)
            except json.JSONDecodeError:
                return AddMemoryResponse.failure(
                    error="Invalid relationships_json format",
                    code=400
                )
        
        if not isinstance(relationships_json, list):
            return AddMemoryResponse.failure(
                error="relationships_json should be a list",
                code=400
            )

        # Convert relationships_json to RelationshipItem objects
        from models.memory_models import RelationshipItem
        relationship_items = []
        for rel in relationships_json:
            if isinstance(rel, dict):
                # Convert dict to RelationshipItem
                relationship_items.append(RelationshipItem(**rel))
            else:
                # Already a RelationshipItem object
                relationship_items.append(rel)

        # Ensure metadata is not None before using it
        if metadata is None:
            metadata = {}

        metadata['user_id'] = str(end_user_id)
        metadata['workspace_id'] = workspace_id if workspace_id is not None else None
        
        # Log what's in metadata - organization_id and namespace_id should already be there
        # from apply_multi_tenant_scoping_to_memory_request() → memory_request.metadata.model_dump()
        # Note: organization_id and namespace_id are direct fields on metadata, NOT inside customMetadata
        logger.info(f"workspace_id: {workspace_id}")
        logger.info(f"organization_id in metadata: {metadata.get('organization_id') if metadata else None}")
        logger.info(f"namespace_id in metadata: {metadata.get('namespace_id') if metadata else None}")

        # Handle ACL
        tenant_id = metadata.get('tenant_id')
        acl_object_ids = metadata.get('acl_object_ids', [])

        if not additional_user_ids:
            additional_user_ids = acl_object_ids

        logger.info(f"additional_user_ids in handle_add_memory: {additional_user_ids}")

        def merge_acl_lists(existing, new):
            return list(set((existing or []) + (new or [])))
        
        acl_fields = [
            'user_read_access', 'user_write_access',
            'workspace_read_access', 'workspace_write_access',
            'role_read_access', 'role_write_access'
        ]
        # Check if any ACL field is already set and non-empty
        acl_already_set = any(metadata.get(field) for field in acl_fields)

        acl = None
        if postMessageId:
            acl = await User.get_acl_for_postMessage(workspace_id=workspace_id, post_message_id=postMessageId, user_id=end_user_id, additional_user_ids=additional_user_ids)
        elif pageId:
            acl = await User.get_acl_for_post(workspace_id=workspace_id, post_id=pageId, user_id=end_user_id, additional_user_ids=additional_user_ids)
        elif connector:
            acl = await User.get_acl_for_workspace(workspace_id=workspace_id, tenant_id=tenant_id, user_id=end_user_id, additional_user_ids=additional_user_ids)

        # Ensure acl is a dict if not None
        if acl is not None and not isinstance(acl, dict):
            acl = {}

        if acl:
            for field in acl_fields:
                metadata[field] = merge_acl_lists(metadata.get(field), acl.get(field) if acl else [])
        elif not acl_already_set:
            # If no ACLs at all, set to private
            metadata['user_read_access'] = [end_user_id]
            metadata['user_write_access'] = [end_user_id]
            metadata['workspace_read_access'] = []
            metadata['workspace_write_access'] = []
            metadata['role_read_access'] = []
            metadata['role_write_access'] = []
        # If ACLs are already set and no new ACLs, do nothing (preserve)

        if pageId is not None:
            metadata['pageId'] = pageId
        if sourceType is not None:
            metadata['sourceType'] = sourceType
        if sourceUrl is not None:
            metadata['sourceUrl'] = sourceUrl

        logger.info(f"metadata in handle_add_memory: {metadata}")

        # Create the appropriate MemoryItem instance based on the types
        try:
            if memory_type in ['text', 'TextMemoryItem', 'message']:
                memory_item = TextMemoryItem(content, metadata, context)
            elif memory_type == 'code_snippet': 
                memory_item = CodeSnippetMemoryItem(content, metadata, context)
            elif memory_type in ['document', 'DocumentMemoryItem']:
                metadata['url'] = url
                metadata['sourceType'] = 'papr'
                memory_item = DocumentMemoryItem(content, metadata, context)
            elif memory_type == 'webpage':
                memory_item = WebpageMemoryItem(content, metadata, context)
            elif memory_type == 'code_file':
                memory_item = CodeFileMemoryItem(content, metadata, context)
            elif memory_type == 'meeting':
                memory_item = MeetingMemoryItem(content, metadata, context)
            elif memory_type == 'plugin':
                memory_item = PluginMemoryItem(content, metadata, context)
            elif memory_type == 'issue':
                memory_item = IssueMemoryItem(content, metadata, context)
            elif memory_type == 'customer':
                memory_item = CustomerMemoryItem(content, metadata, context)
            else:
                return AddMemoryResponse.failure(
                    error="Invalid memory type",
                    code=400
                )
        except Exception as e:
            logger.error(f"Error creating memory item: {str(e)}")
            return AddMemoryResponse.failure(
                error="Error creating memory item",
                code=400,
                details=str(e)
            )

        logger.info(f"memory_item handle_incoming_memory: {memory_item}")
        
        # --- Resolve external user IDs to internal Parse IDs ---
        developer_user_object_id = None
        logger.info(f"memory_request.metadata inside handle_incoming_memory: {memory_request.metadata}")
        if memory_request.metadata and getattr(memory_request.metadata, 'external_user_id', None):
            async with httpx.AsyncClient() as client:
                user_service = User(developer_user_id)
                updated_metadata, resolved, _ = await user_service.resolve_external_user_ids_to_internal(
                    developer_id=developer_user_id,
                    metadata=memory_request.metadata,
                    httpx_client=client,
                    x_api_key=api_key
                )
                memory_request.metadata = updated_metadata
                developer_user_object_id = resolved.get("developerUser_objectId")
                logger.info(f"resolved: {resolved}")

        # Extract graph generation configuration from new structure
        graph_override = None
        schema_id = None
        simple_schema_mode = False
        property_overrides = None
        memory_policy_dict = None

        # NEW: Check for memory_policy first (new unified API)
        if hasattr(memory_request, 'memory_policy') and memory_request.memory_policy:
            mp = memory_request.memory_policy
            # Convert Pydantic model to dict if needed
            memory_policy_dict = mp.model_dump() if hasattr(mp, 'model_dump') else mp

            # Extract schema_id from memory_policy
            schema_id = memory_policy_dict.get('schema_id')

            # Extract mode and configure accordingly
            mode = memory_policy_dict.get('mode', 'auto')
            if mode == 'structured':
                # Structured mode: developer provides exact nodes
                nodes = memory_policy_dict.get('nodes')
                relationships = memory_policy_dict.get('relationships')
                if nodes:
                    graph_override = {'nodes': nodes, 'relationships': relationships or []}
                    logger.info(f"🎯 STRUCTURED MODE (memory_policy): Using developer-provided graph structure")
            elif mode in ['auto', 'hybrid']:
                # Auto/Hybrid mode: LLM extraction with optional constraints
                logger.info(f"🤖 {mode.upper()} MODE (memory_policy): schema_id={schema_id}")

        # LEGACY: Fall back to graph_generation if memory_policy not provided
        elif memory_request.graph_generation:
            graph_gen = memory_request.graph_generation
            if graph_gen.mode == "manual" and graph_gen.manual:
                graph_override = graph_gen.manual
                logger.info(f"🎯 MANUAL MODE: Using developer-provided graph structure")
            elif graph_gen.mode == "auto" and graph_gen.auto:
                auto_config = graph_gen.auto
                schema_id = auto_config.schema_id
                simple_schema_mode = auto_config.simple_schema_mode
                property_overrides = auto_config.property_overrides
                logger.info(f"🤖 AUTO MODE: schema_id={schema_id}, simple_schema_mode={simple_schema_mode}")
                if property_overrides:
                    logger.info(f"🔧 PROPERTY OVERRIDES: {len(property_overrides)} rules")

        # Resolve schema-level memory_policy if schema_id is provided
        if schema_id:
            try:
                resolved_policy = await resolve_memory_policy_from_schema(
                    memory_graph=memory_graph,
                    schema_id=schema_id,
                    memory_policy=memory_policy_dict,
                    user_id=end_user_id,
                    workspace_id=workspace_id,
                    organization_id=metadata.get('organization_id'),
                    namespace_id=metadata.get('namespace_id'),
                    api_key=api_key
                )
                logger.info(f"📋 RESOLVED POLICY: {resolved_policy}")

                # Check if we should skip graph extraction (consent='none')
                if should_skip_graph_extraction(resolved_policy):
                    logger.warning(f"⚠️ Skipping graph extraction due to consent='none' policy")
                    # Continue with memory storage, but skip graph generation
                    graph_override = {'nodes': [], 'relationships': []}

                # Extract OMO fields and add to metadata
                omo_fields = extract_omo_fields_from_policy(resolved_policy)
                metadata['consent'] = omo_fields.get('consent', 'implicit')
                metadata['risk'] = omo_fields.get('risk', 'none')
                if omo_fields.get('omo_acl'):
                    metadata['omo_acl'] = omo_fields['omo_acl']

                # Extract node_constraints for graph processing
                node_constraints = resolved_policy.get('node_constraints')
                if node_constraints:
                    if not property_overrides:
                        property_overrides = []
                    # Convert node_constraints to property_overrides format for compatibility
                    for constraint in node_constraints:
                        if constraint.get('force'):
                            property_overrides.append({
                                'node_type': constraint['node_type'],
                                'properties': constraint['force']
                            })
                    logger.info(f"🔧 NODE CONSTRAINTS from policy: {len(node_constraints)} rules")

            except Exception as e:
                logger.warning(f"Failed to resolve schema-level policy: {e}")
                # Continue without schema policy

        logger.info(f"🔍 DEBUG: Extracted graph_override: {graph_override is not None}")
        logger.info(f"🔍 DEBUG: Extracted schema_id: {schema_id}")
        logger.info(f"🔍 DEBUG: Extracted simple_schema_mode: {simple_schema_mode}")
        logger.info(f"🔍 DEBUG: Extracted property_overrides: {property_overrides}")
        try:
            memory_items = await memory_graph.add_memory_item_async(
                memory_item,
                relationship_items,
                sessionToken,
                end_user_id,
                background_tasks,
                neo_session,
                True,
                workspace_id,
                skip_background_processing,
                user_workspace_ids,
                api_key,
                developer_user_object_id,
                legacy_route=legacy_route,
                developer_user_id=developer_user_id,  # Pass developer ID for schema selection
                graph_override=graph_override,  # Pass extracted graph_override for bypassing LLM
                schema_id=schema_id,  # Pass extracted schema_id for enforcement
                simple_schema_mode=simple_schema_mode, # Pass extracted simple_schema_mode
                property_overrides=property_overrides # Pass extracted property_overrides
            )
        except Exception as e:
            logger.error(f"Error adding memory item to graph: {str(e)}")
            return AddMemoryResponse.failure(
                error="There was an error adding the memory item",
                code=404,
                details=str(e)
            )
        logger.info(f"handle_incoming_memory - Raw memory_items: {memory_items}")
        logger.info(f"handle_incoming_memory - memory_items type: {type(memory_items)}")

        if not memory_items:
            return AddMemoryResponse.failure(
                error="There was an error adding the memory item",
                code=404
            )

        # Use the first memory item for logging and response
        first_memory = memory_items[0] if memory_items else None
        logger.info(f"handle_incoming_memory - First item memoryChunkIds: {first_memory.memoryChunkIds if first_memory else 'No memory items'}")

        # memoryChunkIds are already correctly set by add_memory_item_async
        # No need to overwrite them here - they are in the format [baseId_0, baseId_1, ...]
        
        # Return the memory item data in AddMemoryResponse format
        return AddMemoryResponse.success(
            data=[
                AddMemoryItem(
                    memoryId=item.memoryId,
                    createdAt=item.createdAt,
                    objectId=item.objectId,
                    memoryChunkIds=item.memoryChunkIds
                ) for item in memory_items
            ]
        )

    except Exception as e:
        logger.error(f"Error processing memory item handle_incoming_memory: {str(e)}")
        # Only raise HTTPException for truly unexpected server errors
        raise HTTPException(status_code=500, detail=str(e))
