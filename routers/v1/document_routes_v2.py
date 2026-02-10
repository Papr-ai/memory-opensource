"""
Enhanced document processing routes with pluggable architecture
Integrates TensorLake, Reducto, and Gemini Vision providers with Temporal workflows
"""

from fastapi import APIRouter, HTTPException, Request, Depends, Response, Form, File, UploadFile, BackgroundTasks, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
from typing import Optional, Dict, Any
import uuid
import os
from datetime import datetime
from dotenv import find_dotenv, load_dotenv
from pydantic import AliasChoices

# Core imports
from models.parse_server import DocumentUploadResponse, DocumentUploadStatus, DocumentUploadStatusType
from models.shared_types import MemoryMetadata, PreferredProvider
from models.temporal_models import SchemaSpecification
from memory.memory_graph import MemoryGraph
def _parse_metadata(
    metadata: Optional[str] = Form(None),
) -> Optional[MemoryMetadata]:
    if not metadata:
        return None
    try:
        return MemoryMetadata.model_validate_json(metadata)
    except Exception:
        try:
            return MemoryMetadata(**(_json.loads(metadata) or {}))
        except Exception:
            return None

import json as _json
# SchemaSpecificationMixin import removed - using dictionary instead to avoid Temporal sandbox restrictions
from services.auth_utils import get_user_from_token_optimized
from services.multi_tenant_utils import extract_multi_tenant_context, apply_multi_tenant_scoping_to_metadata
from services.logger_singleton import LoggerSingleton
from services.utils import log_amplitude_event, get_memory_graph

# Document processing imports
from core.document_processing.security import FileValidator
from core.document_processing.provider_manager import TenantConfigManager, DocumentProcessorFactory
from core.document_processing.websocket_manager import get_websocket_manager

# Temporal workflow import
try:
    from cloud_plugins.temporal.client import get_temporal_client
    from cloud_plugins.temporal.workflows.document_processing import DocumentProcessingWorkflow
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    get_temporal_client = None
    DocumentProcessingWorkflow = None

# Config
from config import get_features

logger = LoggerSingleton.get_logger(__name__)

# Load environment variables from .env if present (conditionally based on USE_DOTENV)
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    _ENV_FILE = find_dotenv()
    if _ENV_FILE:
        load_dotenv(_ENV_FILE)

# Security schemes
bearer_auth = HTTPBearer(scheme_name="Bearer", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
session_token_header = APIKeyHeader(name="X-Session-Token", auto_error=False)

router = APIRouter(prefix="/document", tags=["Document"])


@router.post("",
    response_model=DocumentUploadResponse,
    responses={
        200: {"model": DocumentUploadResponse, "description": "Document upload started"},
        202: {"model": DocumentUploadResponse, "description": "Document processing queued"},
        400: {"model": DocumentUploadResponse, "description": "Bad request"},
        401: {"model": DocumentUploadResponse, "description": "Unauthorized"},
        413: {"model": DocumentUploadResponse, "description": "Payload too large"},
        500: {"model": DocumentUploadResponse, "description": "Internal server error"}
    },
    description="""
    Upload and process documents using the pluggable architecture.

    **Authentication Required**: Bearer token or API key

    **Supported Providers**: TensorLake.ai, Reducto AI, Gemini Vision (fallback)

    **Features**:
    - Multi-tenant organization/namespace scoping
    - Temporal workflow for durable execution
    - Real-time WebSocket status updates
    - Integration with Parse Server (Post/PostSocial/PageVersion)
    - Automatic fallback between providers
    """
)
async def upload_document(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    # Document fields (sent via multipart form)
    metadata: Optional[MemoryMetadata] = Depends(_parse_metadata),
    preferred_provider: Optional[PreferredProvider] = Form(None),
    hierarchical_enabled: bool = Form(True),
    schema_id: Optional[str] = Form(None),
    graph_override: Optional[str] = Form(None),
    property_overrides: Optional[str] = Form(None),
    # Unified memory_policy parameter (new API - recommended)
    memory_policy: Optional[str] = Form(
        None,
        description="JSON-encoded memory policy. Includes mode ('auto'/'manual'), "
                   "schema_id, node_constraints (applied in auto mode when present), "
                   "and OMO fields (consent, risk, acl). "
                   "This is the recommended way to configure memory processing."
    ),
    namespace_id: Optional[str] = Form(
        None,
        validation_alias=AliasChoices("namespace_id", "namespace")
    ),
    # User identification (external_user_id is primary, end_user_id is alias for backwards compat)
    external_user_id: Optional[str] = Form(
        None,
        description="Your application's user identifier. This is the primary way to identify users. "
                    "Also accepts legacy 'end_user_id'.",
        validation_alias=AliasChoices("external_user_id", "end_user_id")
    ),
    user_id: Optional[str] = Form(None, description="DEPRECATED: Internal Papr Parse user ID. Most developers should use external_user_id."),
    webhook_url: Optional[str] = Form(None),
    webhook_secret: Optional[str] = Form(None),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    api_key: Optional[str] = Depends(api_key_header),
    session_token: Optional[str] = Depends(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
) -> DocumentUploadResponse:
    """
    Upload and process document using new pluggable architecture.

    Args:
        file: The document file to upload (PDF, DOCX, etc.)
        metadata: JSON string for MemoryMetadata
        preferred_provider: Document processing provider
        hierarchical_enabled: Enable hierarchical extraction
        schema_id: Schema ID for graph extraction
        graph_override: JSON graph override
        property_overrides: JSON property overrides
        memory_policy: JSON-encoded unified memory policy (recommended). Includes mode,
            schema_id, node_constraints, and OMO fields (consent, risk, acl).
        namespace_id: Optional namespace ID for multi-tenancy (accepts legacy 'namespace')
        external_user_id: Your application's user identifier (primary method, accepts legacy end_user_id)
        user_id: DEPRECATED - Internal Papr Parse user ID. Use external_user_id instead.
        webhook_url: Optional webhook for completion notification
        webhook_secret: Optional webhook authentication secret
    """

    upload_id = str(uuid.uuid4())
    try:
        # Authenticate user using optimized auth (mirror memory_routes behavior)
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')

        try:
            # Use the injected singleton memory graph
            # Handle user identification with external_user_id as primary (legacy end_user_id accepted)
            # Log deprecation warnings for old parameter names
            effective_external_user_id = external_user_id
            if user_id:
                logger.warning(
                    f"DEPRECATION WARNING: 'user_id' form parameter is deprecated for document uploads. "
                    f"Use 'external_user_id' instead. Provided: {user_id[:20]}..."
                )

            # Inject user parameters into metadata for auth resolution
            # The auth system expects these in metadata.user_id and metadata.external_user_id
            if metadata:
                if user_id:
                    metadata.user_id = user_id
                if effective_external_user_id:
                    metadata.external_user_id = effective_external_user_id
            elif user_id or effective_external_user_id:
                # Create metadata if it doesn't exist but we have user parameters
                metadata = MemoryMetadata()
                if user_id:
                    metadata.user_id = user_id
                if effective_external_user_id:
                    metadata.external_user_id = effective_external_user_id

            # Create a minimal AddMemoryRequest-like object for auth resolution
            # This allows the auth system to resolve user_id/external_user_id correctly
            from models.memory_models import AddMemoryRequest
            temp_memory_request = AddMemoryRequest(
                content="",  # Dummy content (not used for auth)
                type="text",
                metadata=metadata
            ) if metadata else None

            # Build auth scheme precedence similar to /batch
            import httpx as _httpx
            async with _httpx.AsyncClient() as httpx_client:
                if api_key and bearer_token:
                    # Developer provides API key + Bearer for end user
                    token_hdr = f"Bearer {bearer_token.credentials}"
                elif api_key and session_token:
                    token_hdr = f"Session {session_token}"
                elif api_key:
                    token_hdr = f"APIKey {api_key}"
                elif bearer_token:
                    token_hdr = f"Bearer {bearer_token.credentials}"
                elif session_token:
                    token_hdr = f"Session {session_token}"
                else:
                    token_hdr = request.headers.get('Authorization')

                if not token_hdr:
                    response.status_code = 401
                    return DocumentUploadResponse.failure(
                        error="Missing authentication",
                        code=401,
                        message="Authorization header, X-API-Key, or X-Session-Token required"
                    )

                auth_response = await get_user_from_token_optimized(
                    token_hdr,
                    client_type,
                    memory_graph,
                    api_key=api_key,
                    search_request=None,
                    memory_request=temp_memory_request,
                    batch_request=None,
                    httpx_client=httpx_client
                )

            # Extract identities (mirror add_memory_batch_v1 semantics)
            developer_user_id = auth_response.developer_id
            end_user_id_resolved = auth_response.end_user_id
            user_info = auth_response.user_info
            workspace_id = auth_response.workspace_id
            updated_metadata = auth_response.updated_metadata
            # Fallback to TEST_WORKSPACE_ID from env for testing/e2e
            if not workspace_id:
                workspace_id = os.environ.get("TEST_WORKSPACE_ID") or os.environ.get("DEFAULT_WORKSPACE_ID")
            sessionTokenResolved = getattr(auth_response, 'session_token', None)
            apiKeyResolved = getattr(auth_response, 'api_key', api_key)

            # Use developer_id from user_info if available
            if user_info and isinstance(user_info, dict) and user_info.get("developer_id"):
                developer_user_id = user_info.get("developer_id")
            
            # Use updated_metadata from optimized authentication if available
            # This ensures user_id and external_user_id are properly resolved
            if updated_metadata:
                metadata = updated_metadata
                logger.info(f"Using updated metadata from auth - user_id: {metadata.user_id}, external_user_id: {metadata.external_user_id}")
        except Exception as e:
            response.status_code = 401
            return DocumentUploadResponse.failure(
                error="Authentication failed",
                code=401,
                message=str(e)
            )

        # Check interaction limits (0 mini interactions for document upload - charged via document ingestion fees)
        from models.operation_types import MemoryOperationType
        from services.user_utils import User
        from config.features import get_features
        from os import environ as env
        
        features = get_features()
        if features.is_cloud and not env.get('EVALMETRICS', 'False').lower() == 'true':
            # Extract API key ID from auth_response if available
            api_key_id = None
            if auth_response.api_key_info and isinstance(auth_response.api_key_info, dict):
                api_key_doc = auth_response.api_key_info.get('api_key_doc')
                if api_key_doc:
                    api_key_id = api_key_doc.get('_id') or api_key_doc.get('objectId')
            
            # Extract organization_id and namespace_id from auth_response
            organization_id = auth_response.organization_id if hasattr(auth_response, 'organization_id') else None
            namespace_id = auth_response.namespace_id if hasattr(auth_response, 'namespace_id') else None
            
            user_instance = User(id=developer_user_id)
            # Use the injected singleton memory graph
            limit_check = await user_instance.check_interaction_limits_fast(
                interaction_type='mini',
                memory_graph=memory_graph,
                operation=MemoryOperationType.UPLOAD_DOCUMENT,
                api_key_id=api_key_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            # For zero-cost operations, this will return None (early return)
            # But we still check in case of subscription issues
            if limit_check:
                response_dict, status_code, is_error = limit_check
                if is_error:
                    response.status_code = status_code
                    return DocumentUploadResponse.failure(
                        error=response_dict.get('error'),
                        code=status_code,
                        message=response_dict.get('message')
                    )

        # Extract multi-tenant context
        auth_context = extract_multi_tenant_context(auth_response)
        organization_id = auth_context.get('organization_id')
        namespace_id = namespace_id or auth_context.get('namespace_id')

        logger.info(f"Document upload started - upload_id: {upload_id}, org: {organization_id}, namespace: {namespace_id}")

        # Read file
        file_content = await file.read()

        # Validate content size (similar to add_memory_v1) with a larger default for files
        import os as _os
        # Use MAX_FILE_SIZE_BYTES if set, otherwise default to 50MB (52428800)
        max_content_length_str = _os.environ.get("MAX_FILE_SIZE_BYTES", "52428800")
        if isinstance(max_content_length_str, str):
            max_content_length_str = max_content_length_str.split('#')[0].strip()
        try:
            max_content_length = int(max_content_length_str)
        except Exception:
            max_content_length = 52428800

        if len(file_content) > max_content_length:
            response.status_code = 413
            return DocumentUploadResponse.failure(
                error=f"Content size ({len(file_content)} bytes) exceeds maximum limit of {max_content_length} bytes",
                code=413
            )

        # Validate file security
        logger.info(f"🔒 Validating file: {file.filename}, size: {len(file_content)} bytes")
        is_valid, validation_error = await FileValidator.validate_file(
            file_content, file.filename
        )
        logger.info(f"🔒 Validation result: is_valid={is_valid}, error={validation_error}")

        if not is_valid:
            logger.warning(f"🔒 File validation failed for {file.filename}: {validation_error}")
            response.status_code = 400
            return DocumentUploadResponse.failure(
                error=validation_error,
                code=400,
                message=f"File validation failed: {validation_error}"
            )

        # Build objects from form fields

        # Schema fields - schema_id is just a string now
        # No parsing needed since it's already a string from Form()

        if graph_override:
            try:
                graph_override_obj = _json.loads(graph_override)
            except Exception:
                graph_override_obj = None
        else:
            graph_override_obj = None

        if property_overrides:
            try:
                property_overrides_obj = _json.loads(property_overrides)
            except Exception:
                property_overrides_obj = None
        else:
            property_overrides_obj = None

        # Parse memory_policy (new unified API)
        memory_policy_obj = None
        if memory_policy:
            try:
                memory_policy_obj = _json.loads(memory_policy)
                logger.info(f"📋 Parsed memory_policy: mode={memory_policy_obj.get('mode')}, schema_id={memory_policy_obj.get('schema_id')}")
            except Exception as e:
                logger.warning(f"Failed to parse memory_policy JSON: {e}")
                memory_policy_obj = None

        logger.info(f"📋 Parsed upload request - schema_id: {schema_id}, metadata: {metadata}")
        logger.info(f"📋 Document processing - preferred_provider: {preferred_provider}, hierarchical_enabled: {hierarchical_enabled}, namespace: {namespace_id}")
        
        # Apply multi-tenant scoping to metadata if present
        if metadata:
            try:
                metadata = apply_multi_tenant_scoping_to_metadata(
                    metadata,
                    auth_context
                )
            except Exception as e:
                logger.warning(f"Failed to apply multi-tenant scoping: {e}")
        else:
            # Create empty metadata with multi-tenant scoping
            metadata = MemoryMetadata()
            try:
                metadata = apply_multi_tenant_scoping_to_metadata(
                    metadata,
                    auth_context
                )
            except Exception as e:
                logger.warning(f"Failed to apply multi-tenant scoping to empty metadata: {e}")

        # Store file in Parse Server first for durable processing
        parse_file_url = await _store_file_in_parse_server(
            file_content,
            file.filename,
            end_user_id_resolved,
            workspace_id,
            upload_id,
            sessionTokenResolved or "",
            apiKeyResolved
        )
        # Enrich metadata with file pointer info for downstream Parse models and identities
        if metadata:
            try:
                # Add file info to customMetadata
                if not metadata.customMetadata:
                    metadata.customMetadata = {}
                metadata.customMetadata["file_url"] = parse_file_url
                metadata.customMetadata["file_name"] = file.filename
                metadata.customMetadata["upload_id"] = upload_id
                
                # ALSO set upload_id at top level of metadata (for Memory.upload_id field)
                # This ensures queries for Memory.upload_id work correctly
                metadata.upload_id = upload_id
                
                # Add schema info if provided (for downstream processing)
                if schema_id:
                    metadata.customMetadata["schema_id"] = schema_id

                # Set ACLs based on resolved user identities
                # The auth system has already set metadata.user_id and metadata.external_user_id correctly
                # Now we just need to ensure ACLs are set
                if metadata.external_user_id:
                    # Case: Developer has a separate end user (external_user_id is set)
                    # Add end_user_id to external_user_*_access lists
                    if not metadata.external_user_read_access:
                        metadata.external_user_read_access = []
                    if not metadata.external_user_write_access:
                        metadata.external_user_write_access = []
                    
                    if metadata.external_user_id not in metadata.external_user_read_access:
                        metadata.external_user_read_access.append(metadata.external_user_id)
                    if metadata.external_user_id not in metadata.external_user_write_access:
                        metadata.external_user_write_access.append(metadata.external_user_id)
                    
                    logger.info(f"📋 ACLs (external user): external_user_read_access={metadata.external_user_read_access}, external_user_write_access={metadata.external_user_write_access}")
                elif metadata.user_id:
                    # Case: Developer is the end user (no external_user_id, only user_id)
                    # Add user_id to user_read_access/user_write_access
                    if not metadata.user_read_access:
                        metadata.user_read_access = []
                    if not metadata.user_write_access:
                        metadata.user_write_access = []
                    
                    if metadata.user_id not in metadata.user_read_access:
                        metadata.user_read_access.append(metadata.user_id)
                    if metadata.user_id not in metadata.user_write_access:
                        metadata.user_write_access.append(metadata.user_id)
                    
                    logger.info(f"📋 ACLs (internal user): user_read_access={metadata.user_read_access}, user_write_access={metadata.user_write_access}")
                
                logger.info(f"📋 Enriched metadata with file info, schema info, user identities, and ACLs")
                logger.info(f"📋 Final user identities: user_id={metadata.user_id}, external_user_id={metadata.external_user_id}")
            except Exception as e:
                logger.warning(f"Failed to enrich metadata: {e}")

        # Check if Temporal is available for durable processing
        features = get_features()
        use_temporal = TEMPORAL_AVAILABLE and features.is_enabled("temporal")

        if use_temporal:
            try:
                # Start Temporal workflow for durable processing
                temporal_client = await get_temporal_client()

                workflow_id = f"document-processing-{upload_id}"
                task_queue = "document-processing-v2"  # Use v2 to avoid stuck workflows from previous runs

                # Create file reference for Temporal workflow
                file_reference = {
                    "upload_id": upload_id,
                    "file_url": parse_file_url,
                    "file_name": file.filename,
                    "file_size": len(file_content),
                    "content_type": file.content_type or "application/octet-stream"
                }

                # Coerce provider if a raw string came through
                preferred_provider_enum = None
                try:
                    if isinstance(preferred_provider, str):
                        preferred_provider_enum = PreferredProvider(preferred_provider.lower())
                    else:
                        preferred_provider_enum = preferred_provider
                except Exception:
                    preferred_provider_enum = None

                # Prepare schema specification for workflow as dictionary
                # memory_policy values take precedence over individual parameters
                schema_specification = None
                if schema_id or graph_override_obj or property_overrides_obj or memory_policy_obj:
                    # Start with individual parameters
                    schema_specification = {
                        "schema_id": schema_id,
                        "graph_override": graph_override_obj,
                        "property_overrides": property_overrides_obj,
                        # Default values for new fields
                        "mode": "auto",
                        "node_constraints": None,
                        "consent": "implicit",
                        "risk": "none",
                        "acl": None
                    }

                    # Merge memory_policy if provided (takes precedence)
                    if memory_policy_obj:
                        # Extract values from memory_policy
                        mp_mode = memory_policy_obj.get("mode", "auto")
                        mp_schema_id = memory_policy_obj.get("schema_id")
                        mp_node_constraints = memory_policy_obj.get("node_constraints")
                        mp_consent = memory_policy_obj.get("consent", "implicit")
                        mp_risk = memory_policy_obj.get("risk", "none")
                        mp_acl = memory_policy_obj.get("acl")

                        # Handle structured mode - extract nodes/relationships into graph_override
                        if mp_mode in ("manual", "structured"):  # Accept both, 'structured' is deprecated
                            mp_nodes = memory_policy_obj.get("nodes")
                            mp_relationships = memory_policy_obj.get("relationships")
                            if mp_nodes:
                                schema_specification["graph_override"] = {
                                    "nodes": mp_nodes,
                                    "relationships": mp_relationships or []
                                }

                        # Override with memory_policy values
                        schema_specification["mode"] = mp_mode
                        if mp_schema_id:
                            schema_specification["schema_id"] = mp_schema_id
                        if mp_node_constraints:
                            schema_specification["node_constraints"] = mp_node_constraints
                        schema_specification["consent"] = mp_consent
                        schema_specification["risk"] = mp_risk
                        if mp_acl:
                            schema_specification["acl"] = mp_acl

                        logger.info(
                            f"🚀 Schema specification from memory_policy: mode={mp_mode}, "
                            f"schema_id={schema_specification['schema_id']}, "
                            f"node_constraints={mp_node_constraints is not None}, "
                            f"consent={mp_consent}, risk={mp_risk}"
                        )
                    else:
                        logger.info(
                            f"🚀 Schema specification for workflow: schema_id={schema_id}, "
                            f"graph_override={graph_override_obj is not None}, "
                            f"property_overrides={property_overrides_obj is not None}"
                        )
                
                await temporal_client.start_workflow(
                    DocumentProcessingWorkflow.run,
                    args=[
                        upload_id,
                        organization_id,
                        namespace_id,
                        file_reference,  # File reference instead of content
                        end_user_id_resolved,
                        workspace_id,
                        metadata,  # Use validated MemoryMetadata object
                        preferred_provider_enum,
                        webhook_url,
                        webhook_secret,
                        hierarchical_enabled,
                        schema_specification  # Pass full schema specification for memory indexing
                    ],
                    id=workflow_id,
                    task_queue=task_queue  # Use the v2 task queue
                )

                # Return immediate response with workflow ID
                document_status = DocumentUploadStatus(
                    progress=0.0,
                    upload_id=upload_id,
                    status_type=DocumentUploadStatusType.PROCESSING
                )

                response.status_code = 202
                return DocumentUploadResponse.success(
                    document_status=document_status,
                    memory_items=[],
                    code=202,
                    message="Document processing started with Temporal workflow",
                    details={"workflow_id": workflow_id, "use_temporal": True}
                )

            except Exception as e:
                logger.error(f"Failed to start Temporal workflow: {e}")
                # Fall back to background processing
                use_temporal = False

        if not use_temporal:
            # Process with background tasks (for smaller files or when Temporal unavailable)
            try:
                # Get memory graph with fallback handling
                try:
                    processing_memory_graph = get_memory_graph(request)
                except (AttributeError, KeyError):
                    # During testing, use the same memory_graph from auth
                    processing_memory_graph = memory_graph

                # Create processor factory
                config_manager = TenantConfigManager(processing_memory_graph)
                factory = DocumentProcessorFactory(config_manager)

                # Process document immediately for small files
                processor = await factory.create_processor(
                    organization_id or "default",
                    namespace_id,
                    preferred_provider
                )

                # Quick processing for small files
                async def progress_callback(upload_id, status, progress, current_page, total_pages):
                    # Send WebSocket update
                    try:
                        ws_manager = get_websocket_manager()
                        await ws_manager.broadcast_status_update(
                            {
                                "upload_id": upload_id,
                                "status": status.value if hasattr(status, 'value') else str(status),
                                "progress": progress,
                                "current_page": current_page,
                                "total_pages": total_pages,
                                "timestamp": datetime.now()
                            },
                            organization_id or "default",
                            namespace_id
                        )
                    except Exception as e:
                        logger.warning(f"Failed to send WebSocket update: {e}")

                result = await processor.process_document(
                    file_content,
                    file.filename,
                    upload_id,
                    progress_callback
                )

                # Create success response
                document_status = DocumentUploadStatus(
                    progress=1.0,
                    upload_id=upload_id,
                    status_type=DocumentUploadStatusType.COMPLETED
                )

                # Also persist to Parse Server only (memory handled by Parse Server hooks)
                from cloud_plugins.temporal.activities.document_activities import store_in_parse_only
                storage_result = await store_in_parse_only(
                    processing_result={
                        "pages": [p.model_dump() for p in result.pages],
                        "stats": {
                            "total_pages": result.total_pages,
                            "processing_time": result.processing_time,
                            "confidence": result.confidence,
                            "provider": processor.provider_name
                        },
                        "metadata": result.metadata,
                        "provider_specific": result.provider_specific
                    },

                    metadata=metadata.model_dump() if metadata else {},
                    upload_id=upload_id,
                    user_id=user_id,
                    organization_id=organization_id,
                    namespace_id=namespace_id,
                    workspace_id=auth_response.workspace_id
                )

                response.status_code = 200
                return DocumentUploadResponse.success(
                    document_status=document_status,
                    memory_items=storage_result.get("memory_items", []),
                    code=200,
                    message="Document processed successfully",
                    details={
                        "provider": processor.provider_name,
                        "pages_processed": result.total_pages,
                        "confidence": result.confidence,
                        "use_temporal": False,
                        "parse_records": storage_result.get("parse_records", {})
                    }
                )

            except Exception as e:
                logger.error(f"Background processing failed: {e}")
                response.status_code = 500
                return DocumentUploadResponse.failure(
                    error="Processing failed",
                    code=500,
                    message=str(e)
                )

        # Log to analytics
        try:
            await log_amplitude_event(
                event_type="upload_document",
                user_info=user_info,
                client_type=client_type,
                extra_properties={
                    'upload_id': upload_id,
                    'organization_id': organization_id,
                    'namespace_id': namespace_id,
                    'preferred_provider': preferred_provider,
                    'use_temporal': use_temporal,
                    'file_size': len(file_content)
                },
                end_user_id=user_id
            )
        except Exception as e:
            logger.error(f"Failed to log analytics event: {e}")

    except Exception as e:
        logger.error(f"Document upload failed: {e}", exc_info=True)

        document_status = DocumentUploadStatus(
            progress=0.0,
            upload_id=upload_id,
            status_type=DocumentUploadStatusType.FAILED,
            error=str(e)
        )

        response.status_code = 500
        return DocumentUploadResponse.failure(
            error=str(e),
            code=500,
            message="Internal server error",
            document_status=document_status
        )


@router.get("/status/{upload_id}")
async def get_document_status(
    upload_id: str,
    request: Request,
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    api_key: Optional[str] = Depends(api_key_header),
    session_token: Optional[str] = Depends(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
) -> Dict[str, Any]:
    """Get processing status for an uploaded document"""

    try:
        # Authenticate user (prefer X-API-Key; coerce Bearer API key to APIKey scheme)
        raw_auth_header = request.headers.get('Authorization')

        # Prefer explicit X-API-Key if present
        if api_key:
            auth_header = f"APIKey {api_key}"
        elif session_token:
            auth_header = f"Session {session_token}"
        elif bearer_token:
            bearer_value = bearer_token.credentials
            # Heuristic: if not a JWT (no two dots), treat Bearer value as API key
            if bearer_value.count('.') >= 2:
                auth_header = f"Bearer {bearer_value}"
            else:
                auth_header = f"APIKey {bearer_value}"
        else:
            # Fallback to raw Authorization if provided
            auth_header = raw_auth_header
            # If it's Bearer but looks like an API key, coerce to APIKey
            if auth_header and auth_header.startswith('Bearer '):
                token_val = auth_header.split(' ', 1)[1].strip()
                if token_val and token_val.count('.') < 2:
                    auth_header = f"APIKey {token_val}"

        if not auth_header:
            raise HTTPException(status_code=401, detail="Authorization header, X-API-Key, or X-Session-Token required")

        # Use the injected singleton memory graph
        auth_response = await get_user_from_token_optimized(auth_header, "api", memory_graph)

        auth_context = extract_multi_tenant_context(auth_response)
        organization_id = auth_context.get('organization_id')

        # Check if this is a Temporal workflow
        if TEMPORAL_AVAILABLE:
            try:
                temporal_client = await get_temporal_client()
                handle = temporal_client.get_workflow_handle(f"document-processing-{upload_id}")

                # Query workflow status
                status = await handle.query(DocumentProcessingWorkflow.get_status)

                return {
                    "upload_id": upload_id,
                    "status": status["status"],
                    "progress": status["progress"],
                    "current_page": status.get("current_page"),
                    "total_pages": status.get("total_pages"),
                    "error": status.get("error"),
                    "timestamp": status.get("timestamp"),
                    "page_id": status.get("page_id"),  # User-facing Post ID
                    "workflow_type": "temporal"
                }

            except Exception:
                pass

        # Not a Temporal workflow or Temporal not available, check other storage
        # This would query status from database/cache
        return {
            "upload_id": upload_id,
            "status": "unknown",
            "message": "Status not found",
            "workflow_type": "background"
        }

    except Exception as e:
        logger.error(f"Failed to get document status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{upload_id}")
async def cancel_document_processing(
    upload_id: str,
    request: Request,
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    api_key: Optional[str] = Depends(api_key_header),
    session_token: Optional[str] = Depends(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
) -> Dict[str, Any]:
    """Cancel document processing"""

    try:
        # Authenticate user (prefer X-API-Key; coerce Bearer API key to APIKey scheme)
        raw_auth_header = request.headers.get('Authorization')

        # Prefer explicit X-API-Key if present
        if api_key:
            auth_header = f"APIKey {api_key}"
        elif session_token:
            auth_header = f"Session {session_token}"
        elif bearer_token:
            bearer_value = bearer_token.credentials
            if bearer_value.count('.') >= 2:
                auth_header = f"Bearer {bearer_value}"
            else:
                auth_header = f"APIKey {bearer_value}"
        else:
            auth_header = raw_auth_header
            if auth_header and auth_header.startswith('Bearer '):
                token_val = auth_header.split(' ', 1)[1].strip()
                if token_val and token_val.count('.') < 2:
                    auth_header = f"APIKey {token_val}"

        if not auth_header:
            raise HTTPException(status_code=401, detail="Authorization header, X-API-Key, or X-Session-Token required")

        # Use the injected singleton memory graph
        auth_response = await get_user_from_token_optimized(auth_header, "api", memory_graph)

        # Try to cancel Temporal workflow
        if TEMPORAL_AVAILABLE:
            try:
                temporal_client = await get_temporal_client()
                handle = temporal_client.get_workflow_handle(f"document-processing-{upload_id}")

                # Send cancellation signal
                await handle.signal(DocumentProcessingWorkflow.cancel_processing, "User requested cancellation")

                return {
                    "upload_id": upload_id,
                    "status": "cancelled",
                    "message": "Document processing cancelled"
                }

            except Exception as e:
                pass

        return {
            "upload_id": upload_id,
            "status": "not_found",
            "message": "Processing not found or already completed"
        }

    except Exception as e:
        logger.error(f"Failed to cancel document processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _store_file_in_parse_server(
    file_content: bytes,
    filename: str,
    user_id: str,
    workspace_id: str,
    upload_id: str,
    session_token: str = "",
    api_key: Optional[str] = None
) -> str:
    """Store file in Parse Server and return the file URL (raw-bytes upload)."""

    try:
        # Defer imports to keep scope minimal
        import magic  # type: ignore
        from services.memory_management import upload_file_to_parse
        from services.user_utils import PARSE_SERVER_URL, PARSE_APPLICATION_ID, PARSE_MASTER_KEY

        # Validate Parse Server configuration before attempting upload
        if not PARSE_SERVER_URL:
            error_msg = "PARSE_SERVER_URL is not configured"
            logger.error(error_msg)
            raise Exception(f"File storage failed: {error_msg}")
        
        if not PARSE_APPLICATION_ID:
            error_msg = "PARSE_APPLICATION_ID is not configured"
            logger.error(error_msg)
            raise Exception(f"File storage failed: {error_msg}")
        
        if api_key and not PARSE_MASTER_KEY:
            error_msg = "PARSE_MASTER_KEY is not configured (required for API key authentication)"
            logger.error(error_msg)
            raise Exception(f"File storage failed: {error_msg}")

        logger.info(f"Attempting to store file in Parse Server: filename={filename}, size={len(file_content)} bytes, upload_id={upload_id}, parse_url={PARSE_SERVER_URL}")

        # Sanitize filename for Parse Server (removes invalid characters like /, \, :, *, ?, ", <, >, |)
        # Parse Server has strict filename requirements - code 122 error indicates invalid characters
        from werkzeug.utils import secure_filename
        sanitized_filename = secure_filename(filename)
        
        # If sanitization removed everything (unlikely but possible), use a fallback
        if not sanitized_filename or sanitized_filename.strip() == "":
            # Extract extension if present
            _, ext = os.path.splitext(filename)
            sanitized_filename = f"uploaded_file{ext}" if ext else "uploaded_file"
            logger.warning(f"Filename sanitization resulted in empty string, using fallback: {sanitized_filename}")
        
        # Preserve original filename in logs for debugging
        if sanitized_filename != filename:
            logger.info(f"Filename sanitized: '{filename}' -> '{sanitized_filename}'")
        
        # Construct Parse Server filename with upload_id prefix
        parse_filename = f"{upload_id}_{sanitized_filename}"

        # Detect mime type from buffer; default to octet-stream
        try:
            mime_type = magic.from_buffer(file_content, mime=True) or "application/octet-stream"
        except Exception as magic_error:
            logger.warning(f"Failed to detect MIME type with magic: {magic_error}, using default")
            mime_type = "application/octet-stream"

        # Reuse the proven v1 helper which performs raw-byte upload
        # Use master key auth by passing a truthy api_key; session_token not required in this flow
        file_info = await upload_file_to_parse(
            file_content=file_content,
            filename=parse_filename,
            content_type=mime_type,
            session_token=session_token,
            api_key=api_key
        )

        if not file_info:
            error_msg = "Parse Server file upload returned None - check Parse Server logs for details"
            logger.error(f"{error_msg}. Parse URL: {PARSE_SERVER_URL}, App ID: {PARSE_APPLICATION_ID}")
            raise Exception(f"File storage failed: {error_msg}")
        
        if not file_info.file_url:
            error_msg = f"Parse Server file upload succeeded but file_url is missing. Response: {file_info}"
            logger.error(error_msg)
            raise Exception(f"File storage failed: {error_msg}")

        logger.info(f"File stored in Parse Server: {file_info.file_url}")
        return file_info.file_url

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to store file in Parse Server: {error_msg}", exc_info=True)
        # Preserve original error message if it already contains "File storage failed"
        if "File storage failed" in error_msg:
            raise Exception(error_msg)
        else:
            raise Exception(f"File storage failed: {error_msg}")