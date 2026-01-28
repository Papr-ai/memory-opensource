from fastapi import APIRouter, FastAPI, HTTPException, Request, Depends, Response, Form, File, UploadFile, BackgroundTasks, Header, Query, Body, status
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials
from typing import Optional, Dict, Any, List, Union
import json
import httpx
import os
import re
from os import environ as env
from toon import encode as toon_encode
from models.memory_models import SearchResponse, SearchRequest, MemoryMetadata, SearchResult, RelationshipItem, NeoNode, ResponseFormat
from memory.memory_graph import MemoryGraph
from services.auth_utils import get_user_from_token, get_user_from_token_optimized, validate_user_identification
from services.user_utils import User
from routers.v1 import v1_router  # Import the shared v1_router
from services.logger_singleton import LoggerSingleton
from api_handlers.chat_gpt_completion import ChatGPTCompletion
from datetime import datetime, timedelta, UTC, timezone
from fastapi import Request, Security
from models.parse_server import (
    ParseStoredMemory, AddMemoryResponse, AddMemoryOMOResponse, ErrorDetail, DeletionStatus, BatchMemoryResponse, BatchMemoryError, DeleteMemoryResponse, UpdateMemoryResponse, UpdateMemoryItem, SystemUpdateStatus, DocumentUploadResponse, DocumentUploadStatus, AddMemoryItem, Memory, ParsePointer, QueryLog
)
from models.omo import memory_to_omo, should_return_omo_format
from models.memory_models import GetMemoryResponse, SearchResponse, SearchRequest, SearchResult, AddMemoryRequest, BatchMemoryRequest, UpdateMemoryRequest, MemoryMetadata
from pydantic import ValidationError
from services.utils import log_amplitude_event, serialize_datetime, get_memory_graph
from routes.memory_routes import common_add_memory_handler, common_add_memory_batch_handler
from services.memory_management import get_document_upload_status, retrieve_memory_item_by_qdrant_id
from fastapi.security import APIKeyHeader
from fastapi.security import HTTPBearer
from amplitude import Amplitude
from dotenv import find_dotenv, load_dotenv
from os import environ as env
from models.memory_models import RerankingConfig, RerankingProvider
from services.query_log_service import query_log_service
from services.token_utils import count_query_embedding_tokens, count_retrieved_memory_tokens, count_neo_nodes_tokens
import time
import asyncio
import uuid


def strip_empty_values(obj: Any) -> Any:
    """
    Recursively removes empty values from dictionaries and lists.
    Removes: None, empty strings, empty lists, empty dicts
    This helps reduce token usage when sending responses to LLMs.
    """
    if isinstance(obj, dict):
        return {
            k: strip_empty_values(v)
            for k, v in obj.items()
            if v is not None and v != "" and v != [] and v != {}
        }
    elif isinstance(obj, list):
        stripped_list = [strip_empty_values(item) for item in obj]
        # Remove empty items from list
        return [item for item in stripped_list if item is not None and item != "" and item != [] and item != {}]
    else:
        return obj

# Load environment variables (conditionally based on USE_DOTENV)
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    ENV_FILE = find_dotenv()
    if ENV_FILE:
        load_dotenv(ENV_FILE)
# Security schemes
bearer_auth = HTTPBearer(scheme_name="Bearer", bearerFormat="JWT", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
session_token_header = APIKeyHeader(name="X-Session-Token", auto_error=False)


amplitude_client = Amplitude(env.get("AMPLITUDE_API_KEY"))

logger = LoggerSingleton.get_logger(__name__)

router = APIRouter(prefix="/memory", tags=["Memory"])

@router.post("",
    response_model=AddMemoryResponse,
    responses={
        200: {
            "model": AddMemoryResponse,
            "description": "Memory successfully added",
            "content": {
                "application/json": {
                    "example": {
                        "code": 200,
                        "status": "success",
                        "data": [
                            {
                                "id": "mem_123",
                                "content": "Sample memory content.",
                                "type": "text",
                                "metadata": {
                                    "topics": "example, memory",
                                    "role": "user",
                                    "category": "task"
                                },
                                "createdAt": "2024-06-01T12:00:00Z"
                            }
                        ],
                        "error": None,
                        "details": None
                    }
                }
            }
        },
        207: {
            "model": AddMemoryResponse,
            "description": "Memory added with degraded functionality",
            "content": {
                "application/json": {
                    "example": {
                        "code": 207,
                        "status": "success",
                        "data": [
                            {
                                "id": "mem_124",
                                "content": "Sample memory content.",
                                "type": "text",
                                "metadata": {"topics": "example, memory"},
                                "createdAt": "2024-06-01T12:01:00Z"
                            }
                        ],
                        "error": None,
                        "details": {"warning": "Neo4j unavailable, stored in Pinecone only."}
                    }
                }
            }
        },
        400: {
            "model": AddMemoryResponse,
            "description": "Bad request",
            "content": {
                "application/json": {
                    "example": {
                        "code": 400,
                        "status": "error",
                        "data": None,
                        "error": "Invalid request payload.",
                        "details": {"field": "content", "reason": "Missing required field."}
                    }
                }
            }
        },
        401: {
            "model": AddMemoryResponse,
            "description": "Unauthorized",
            "content": {
                "application/json": {
                    "example": {
                        "code": 401,
                        "status": "error",
                        "data": None,
                        "error": "Missing or invalid authentication.",
                        "details": None
                    }
                }
            }
        },
        403: {
            "model": AddMemoryResponse,
            "description": "Subscription limit reached",
            "content": {
                "application/json": {
                    "example": {
                        "code": 403,
                        "status": "error",
                        "data": None,
                        "error": "Subscription limit reached. Upgrade required.",
                        "details": {"limit": "memory quota"}
                    }
                }
            }
        },
        413: {
            "model": AddMemoryResponse,
            "description": "Content too large",
            "content": {
                "application/json": {
                    "example": {
                        "code": 413,
                        "status": "error",
                        "data": None,
                        "error": "Content size (16000 bytes) exceeds maximum limit of 15000 bytes.",
                        "details": {"max_content_length": 15000}
                    }
                }
            }
        },
        415: {
            "model": AddMemoryResponse,
            "description": "Unsupported Media Type",
            "content": {
                "application/json": {
                    "example": {
                        "code": 415,
                        "status": "error",
                        "data": None,
                        "error": "Unsupported media type. Use application/json.",
                        "details": None
                    }
                }
            }
        },
        500: {
            "model": AddMemoryResponse,
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "code": 500,
                        "status": "error",
                        "data": None,
                        "error": "Internal server error.",
                        "details": {"trace_id": "abc123"}
                    }
                }
            }
        }
    },
    description="""Add a new memory item to the system with size validation and background processing.
    
    **Authentication Required**:
    One of the following authentication methods must be used:
    - Bearer token in `Authorization` header
    - API Key in `X-API-Key` header
    - Session token in `X-Session-Token` header
    
    **Required Headers**:
    - Content-Type: application/json
    - X-Client-Type: (e.g., 'papr_plugin', 'browser_extension')
    
    **Role-Based Memory Categories**:
    - **User memories**: preference, task, goal, facts, context
    - **Assistant memories**: skills, learning
    
    **New Metadata Fields**:
    - `metadata.role`: Optional field to specify who generated the memory (user or assistant)
    - `metadata.category`: Optional field for memory categorization based on role
    - Both fields are stored within metadata at the same level as topics, location, etc.
    
    The API validates content size against MAX_CONTENT_LENGTH environment variable (defaults to 15000 bytes).
    """,
    openapi_extra={
        "operationId": "add_memory",
        "x-openai-isConsequential": False
    }
    )
async def add_memory_v1(
    request: Request,
    memory_request: AddMemoryRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Security(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Security(session_token_header),
    skip_background_processing: bool = Query(False, description="If True, skips adding background tasks for processing"),
    enable_holographic: bool = Query(False, description="If True, applies holographic neural transforms and stores in holographic collection"),
    format: Optional[str] = Query(None, description="Response format. Use 'omo' for Open Memory Object standard format (portable across platforms)."),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
) -> Union[AddMemoryResponse, "AddMemoryOMOResponse"]:
    logger.info(f"=== add_memory_v1 called ===")
    logger.info(f"GRAPH_GENERATION_DEBUG: {memory_request.graph_generation}")
    logger.info(f"GRAPH_GENERATION_TYPE: {type(memory_request.graph_generation)}")
    logger.info(f"Request path: {request.url.path}")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Memory request: {memory_request}")
    """
    Add a new memory item to the system with size validation and background processing.
    """
    try:

        # Validate content size
        content_length = len(memory_request.content.encode('utf-8'))
        max_content_length_str = env.get("MAX_CONTENT_LENGTH", "15000")
        if isinstance(max_content_length_str, str):
            max_content_length_str = max_content_length_str.split('#')[0].strip()
        max_content_length = int(max_content_length_str)
        
        if content_length > max_content_length:
            response.status_code = 413
            return AddMemoryResponse.failure(
                error=f"Content size ({content_length} bytes) exceeds maximum limit of {max_content_length} bytes",
                code=413
            )

        auth_header = request.headers.get('Authorization')
        if not auth_header and api_key:
            # Patch the request headers to include Authorization for downstream code
            request.headers.__dict__['_list'].append(
                (b'authorization', f'APIKey {api_key}'.encode())
            )
            auth_header = request.headers.get('Authorization')

        if not auth_header and not api_key and not session_token:
            response.status_code = 401
            return AddMemoryResponse.failure(
                error="Missing authentication",
                code=401
            )

        # --- Optimized authentication using cached method ---
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')
        auth_start_time = time.time()
        auth_response = None
        try:
            async with httpx.AsyncClient() as httpx_client:
                # Use optimized authentication that gets workspace_id, isQwenRoute, user_roles, user_workspace_ids, and resolves user in parallel
                if api_key and bearer_token:
                    # Developer provides API key + Bearer token for end user - use Bearer token for auth but developer's API key for Parse
                    auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", client_type, memory_graph, api_key=api_key, search_request=None, memory_request=memory_request, httpx_client=httpx_client)
                elif api_key and session_token:
                    auth_response = await get_user_from_token_optimized(f"Session {session_token}", client_type, memory_graph, api_key=api_key, search_request=None, memory_request=memory_request, httpx_client=httpx_client)
                elif api_key:
                    auth_response = await get_user_from_token_optimized(f"APIKey {api_key}", client_type, memory_graph, search_request=None, memory_request=memory_request, httpx_client=httpx_client)
                elif bearer_token:
                    auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", client_type, memory_graph, search_request=None, memory_request=memory_request, httpx_client=httpx_client)
                elif session_token:
                    auth_response = await get_user_from_token_optimized(f"Session {session_token}", client_type, memory_graph, search_request=None, memory_request=memory_request, httpx_client=httpx_client)
                else:
                    auth_response = await get_user_from_token_optimized(auth_header, client_type, memory_graph, search_request=None, memory_request=memory_request, httpx_client=httpx_client)
        except ValueError as e:
            logger.error(f"Invalid authentication token: {e}")
            response.status_code = 401
            return AddMemoryResponse.failure(
                error="Invalid authentication token",
                code=401
            )
        except Exception as e:
            logger.error(f"Authentication system error: {e}")
            logger.error("Full traceback:", exc_info=True)
            response.status_code = 500
            return AddMemoryResponse.failure(
                error="Authentication system error",
                code=500
            )
        
        # Defensive check to ensure auth_response is valid
        if not auth_response:
            logger.error("Authentication failed: auth_response is None")
            response.status_code = 500
            return AddMemoryResponse.failure(
                error="Authentication system error",
                code=500
            )

        auth_end_time = time.time()
        logger.info(f"Enhanced authentication timing: {(auth_end_time - auth_start_time) * 1000:.2f}ms")

        # Validate user_id to prevent common mistakes (external ID in user_id field)
        async with httpx.AsyncClient() as httpx_client:
            validation_error = await validate_user_identification(
                memory_request, memory_graph, httpx_client
            )
            if validation_error:
                response.status_code = 400
                return AddMemoryResponse.failure(
                    error=validation_error.reason,
                    code=400,
                    details={
                        "field": validation_error.field,
                        "provided_value": validation_error.provided_value,
                        "suggestion": validation_error.suggestion
                    }
                )

        # Extract values from the optimized auth response
        user_id = auth_response.developer_id
        end_user_id = auth_response.end_user_id
        sessionToken = auth_response.session_token
        user_info = auth_response.user_info
        api_key = auth_response.api_key
        workspace_id = auth_response.workspace_id
        is_qwen_route = auth_response.is_qwen_route
        user_roles = auth_response.user_roles
        user_workspace_ids = auth_response.user_workspace_ids
        updated_metadata = auth_response.updated_metadata

        # Extract multi-tenant context using utility function
        from services.multi_tenant_utils import extract_multi_tenant_context, apply_multi_tenant_scoping_to_memory_request
        auth_context = extract_multi_tenant_context(auth_response)

        # Set developer_id from user_info
        developer_id = user_info.get("developer_id") if user_info and "developer_id" in user_info else user_id

        # Check interaction limits (4 mini interactions for add_memory)
        from models.operation_types import MemoryOperationType
        from config.features import get_features
        features = get_features()
        
        # Initialize these variables before the if block (needed for common_add_memory_handler call)
        api_key_id = None
        organization_id = None
        namespace_id = None
        
        if features.is_cloud and not env.get('EVALMETRICS', 'False').lower() == 'true':
            # Extract API key ID from auth_response if available
            if auth_response.api_key_info and isinstance(auth_response.api_key_info, dict):
                api_key_doc = auth_response.api_key_info.get('api_key_doc')
                if api_key_doc:
                    api_key_id = api_key_doc.get('_id') or api_key_doc.get('objectId')
            
            # Extract organization_id and namespace_id from auth_response
            organization_id = auth_response.organization_id if hasattr(auth_response, 'organization_id') else None
            namespace_id = auth_response.namespace_id if hasattr(auth_response, 'namespace_id') else None
            
            user_instance = User(id=user_id)
            limit_check = await user_instance.check_interaction_limits_fast(
                interaction_type='mini',
                memory_graph=memory_graph,
                operation=MemoryOperationType.ADD_MEMORY,
                api_key_id=api_key_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            if limit_check:
                response_dict, status_code, is_error = limit_check
                if is_error:
                    response.status_code = status_code
                    return AddMemoryResponse.failure(
                        error=response_dict.get('error'),
                        code=status_code,
                        message=response_dict.get('message'),
                        details=response_dict
                    )

        # Use updated_metadata from optimized authentication if available
        if updated_metadata:
            memory_request.metadata = updated_metadata
            logger.info(f"Updated memory_request.metadata with optimized auth data: {memory_request.metadata}")
        elif memory_request.metadata and workspace_id:
            # Fallback: Update metadata with workspace_id if we got it from authentication
            memory_request.metadata.workspace_id = workspace_id
            logger.info(f"Updated memory_request.metadata with workspace_id: {memory_request.metadata}")

        # Apply multi-tenant scoping using utility function
        apply_multi_tenant_scoping_to_memory_request(memory_request, auth_context)

        logger.info(f"Resolved end_user_id: {end_user_id}, developer_id: {developer_id}")
        logger.info(f"is_qwen_route: {is_qwen_route}")
        logger.info(f"workspace_id: {workspace_id}")
        if workspace_id and (not user_workspace_ids or workspace_id not in user_workspace_ids):
            user_workspace_ids = (user_workspace_ids or []) + [workspace_id]
            logger.info(
                "Augmented user_workspace_ids with workspace_id from auth: %s",
                workspace_id
            )
        logger.info(f"memory_request.metadata: {memory_request.metadata}")

        # Normalize is_qwen_route: if None or False, default to False
        if is_qwen_route is None:
            is_qwen_route = False
            logger.info(f"is_qwen_route was None, defaulting to False for add memory route")
        
        # legacy_route is the opposite of is_qwen_route
        legacy_route = not is_qwen_route
        logger.info(f"Setting legacy_route for add memory: {legacy_route} (is_qwen_route: {is_qwen_route})")
         
        await memory_graph.ensure_async_connection()
        async with memory_graph.async_neo_conn.get_session() as neo_session:
            # Use common handler with auth_response
            result = await common_add_memory_handler(
                request,
                memory_graph,
                background_tasks,
                neo_session,
                auth_response,
                memory_request,
                tenant_subtenant=None,  # v1 routes don't use tenant/connector/stream
                connector=None,
                stream=None,
                skip_background_processing=skip_background_processing,
                upload_id=None,  # v1 routes don't use upload_id
                post_objectId=None,  # v1 routes don't use post_objectId
                legacy_route=legacy_route,  # Use the normalized legacy_route
                api_key_id=api_key_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )

        # Always set response.status_code to match the code in the AddMemoryResponse
        if hasattr(result, 'code'):
            response.status_code = result.code
        else:
            response.status_code = 500

        # Add telemetry logging as background task (edition-aware)
        async def log_add_memory_telemetry():
            """Background task for telemetry"""
            try:
                from core.services.telemetry import get_telemetry
                telemetry = get_telemetry()
                await telemetry.track(
                    "add_memory",
                    {
                        "client_type": client_type,
                        "memory_id": result.data[0].memoryId if result.data and len(result.data) > 0 else None,
                        "api_key": api_key,
                    },
                    user_id=end_user_id,
                    developer_id=developer_id
                )
            except Exception as e:
                logger.error(f"Error tracking add_memory: {e}")
        
        background_tasks.add_task(log_add_memory_telemetry)

        # If OMO format requested, convert the response to OMO standard format
        if should_return_omo_format(format) and result.status == "success" and result.data:
            try:
                # Get the first memory item (add_memory returns a single item)
                memory_item = result.data[0]

                # Convert to OMO format using the original request data
                omo_object = memory_to_omo(
                    memory_id=memory_item.memoryId,
                    content=memory_request.content,
                    memory_type=memory_request.type.value if memory_request.type else "text",
                    metadata=memory_request.metadata,
                    memory_policy=memory_request.memory_policy
                )

                # Return OMO format response
                return AddMemoryOMOResponse.success(
                    omo_object=omo_object.model_dump(mode='json'),
                    code=result.code
                )
            except Exception as e:
                logger.error(f"Error converting to OMO format: {e}", exc_info=True)
                # Fall through to normal response if OMO conversion fails

        return result
        
    except Exception as e:
        logger.error(f"Error processing memory: {e}", exc_info=True)
        response.status_code = 500
        return AddMemoryResponse.failure(
            error=str(e),
            code=500
        )

@router.put("/{memory_id}",
    response_model=UpdateMemoryResponse,
    responses={
        200: {"model": UpdateMemoryResponse, "description": "Memory successfully updated"},
        400: {"model": UpdateMemoryResponse, "description": "Bad request"},
        401: {"model": UpdateMemoryResponse, "description": "Unauthorized"},
        404: {"model": UpdateMemoryResponse, "description": "Memory not found"},
        413: {"model": UpdateMemoryResponse, "description": "Content too large"},
        500: {"model": UpdateMemoryResponse, "description": "Internal server error"}
    },
    description="""Update an existing memory item by ID.
    
    **Authentication Required**:
    One of the following authentication methods must be used:
    - Bearer token in `Authorization` header
    - API Key in `X-API-Key` header
    - Session token in `X-Session-Token` header
    
    **Required Headers**:
    - Content-Type: application/json
    - X-Client-Type: (e.g., 'papr_plugin', 'browser_extension')
    
    The API validates content size against MAX_CONTENT_LENGTH environment variable (defaults to 15000 bytes).
    """,
    openapi_extra={
        "operationId": "update_memory",
        "x-openai-isConsequential": True
    }
    )
async def update_memory_v1(
    memory_id: str,
    request: Request,
    update_request: UpdateMemoryRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Security(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Security(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
) -> UpdateMemoryResponse:
    """
    Update an existing memory item by ID.
    """
    try:
        # Validate content size if content is being updated
        if update_request.content is not None:
            content_length = len(update_request.content.encode('utf-8'))
            max_content_length_str = env.get("MAX_CONTENT_LENGTH", "15000")
            if isinstance(max_content_length_str, str):
                max_content_length_str = max_content_length_str.split('#')[0].strip()
            max_content_length = int(max_content_length_str)
            
            if content_length > max_content_length:
                response.status_code = 413
                return UpdateMemoryResponse.failure(
                    error=f"Content size ({content_length} bytes) exceeds maximum limit of {max_content_length} bytes",
                    code=413
                )

        # Get client type
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')

        # Authenticate user
        auth_header = request.headers.get('Authorization')
        if not auth_header and api_key:
            # Patch the request headers to include Authorization for downstream code
            request.headers.__dict__['_list'].append(
                (b'authorization', f'APIKey {api_key}'.encode())
            )
            auth_header = request.headers.get('Authorization')

        if not auth_header and not api_key and not session_token:
            response.status_code = 401
            return UpdateMemoryResponse.failure(
                error="Missing authentication",
                code=401
            )


        # --- Optimized authentication using cached method ---
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')
        auth_start_time = time.time()
        try:
            async with httpx.AsyncClient() as httpx_client:
                # Use optimized authentication that gets workspace_id, isQwenRoute, user_roles, user_workspace_ids, and resolves user in parallel
                if api_key and bearer_token:
                    # Developer provides API key + Bearer token for end user - use Bearer token for auth but developer's API key for Parse
                    auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", client_type, memory_graph, api_key=api_key, search_request=update_request, httpx_client=httpx_client)
                elif api_key and session_token:
                    auth_response = await get_user_from_token_optimized(f"Session {session_token}", client_type, memory_graph, api_key=api_key, search_request=update_request, httpx_client=httpx_client)
                elif api_key:
                    auth_response = await get_user_from_token_optimized(f"APIKey {api_key}", client_type, memory_graph, search_request=update_request, httpx_client=httpx_client)
                elif bearer_token:
                    auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", client_type, memory_graph, search_request=update_request, httpx_client=httpx_client)
                elif session_token:
                    auth_response = await get_user_from_token_optimized(f"Session {session_token}", client_type, memory_graph, search_request=update_request, httpx_client=httpx_client)
                else:
                    auth_response = await get_user_from_token_optimized(auth_header, client_type, memory_graph, search_request=update_request, httpx_client=httpx_client)
        except ValueError as e:
            logger.error(f"Invalid authentication token: {e}")
            response.status_code = 401
            return UpdateMemoryResponse.failure(
                error="Invalid authentication token",
                code=401
            )
        auth_end_time = time.time()
        logger.info(f"Enhanced authentication timing: {(auth_end_time - auth_start_time) * 1000:.2f}ms")

        # Extract values from the optimized auth response
        user_id = auth_response.developer_id
        end_user_id = auth_response.end_user_id
        sessionToken = auth_response.session_token
        user_info = auth_response.user_info
        api_key = auth_response.api_key
        workspace_id = auth_response.workspace_id
        is_qwen_route = auth_response.is_qwen_route
        user_roles = auth_response.user_roles
        user_workspace_ids = auth_response.user_workspace_ids
        updated_metadata = auth_response.updated_metadata
        
        # Set developer_id from user_info
        developer_id = user_info.get("developer_id") if user_info and "developer_id" in user_info else user_id
        
        # Check interaction limits (1 mini interaction for update_memory)
        from models.operation_types import MemoryOperationType
        from config.features import get_features
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
            
            user_instance = User(id=user_id)
            limit_check = await user_instance.check_interaction_limits_fast(
                interaction_type='mini',
                memory_graph=memory_graph,
                operation=MemoryOperationType.UPDATE_MEMORY,
                api_key_id=api_key_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            if limit_check:
                response_dict, status_code, is_error = limit_check
                if is_error:
                    response.status_code = status_code
                    return UpdateMemoryResponse.failure(
                        error=response_dict.get('error'),
                        code=status_code,
                        message=response_dict.get('message'),
                        details=response_dict
                    )
        
        # Use updated_metadata from optimized authentication if available
        if updated_metadata:
            # Preserve ACL fields from the original request while using auth metadata for other fields
            original_metadata = update_request.metadata
            update_request.metadata = updated_metadata
            
            # Preserve ACL fields from the original request
            if original_metadata:
                if hasattr(original_metadata, 'user_read_access') and original_metadata.user_read_access:
                    update_request.metadata.user_read_access = original_metadata.user_read_access
                if hasattr(original_metadata, 'user_write_access') and original_metadata.user_write_access:
                    update_request.metadata.user_write_access = original_metadata.user_write_access
                if hasattr(original_metadata, 'external_user_read_access') and original_metadata.external_user_read_access:
                    update_request.metadata.external_user_read_access = original_metadata.external_user_read_access
                if hasattr(original_metadata, 'external_user_write_access') and original_metadata.external_user_write_access:
                    update_request.metadata.external_user_write_access = original_metadata.external_user_write_access
                if hasattr(original_metadata, 'workspace_read_access') and original_metadata.workspace_read_access:
                    update_request.metadata.workspace_read_access = original_metadata.workspace_read_access
                if hasattr(original_metadata, 'workspace_write_access') and original_metadata.workspace_write_access:
                    update_request.metadata.workspace_write_access = original_metadata.workspace_write_access
                if hasattr(original_metadata, 'role_read_access') and original_metadata.role_read_access:
                    update_request.metadata.role_read_access = original_metadata.role_read_access
                if hasattr(original_metadata, 'role_write_access') and original_metadata.role_write_access:
                    update_request.metadata.role_write_access = original_metadata.role_write_access
                if hasattr(original_metadata, 'namespace_read_access') and original_metadata.namespace_read_access:
                    update_request.metadata.namespace_read_access = original_metadata.namespace_read_access
                if hasattr(original_metadata, 'namespace_write_access') and original_metadata.namespace_write_access:
                    update_request.metadata.namespace_write_access = original_metadata.namespace_write_access
                if hasattr(original_metadata, 'organization_read_access') and original_metadata.organization_read_access:
                    update_request.metadata.organization_read_access = original_metadata.organization_read_access
                if hasattr(original_metadata, 'organization_write_access') and original_metadata.organization_write_access:
                    update_request.metadata.organization_write_access = original_metadata.organization_write_access
            
            logger.info(f"Updated update_request.metadata with optimized auth data (preserving ACL fields): {update_request.metadata}")
        elif update_request.metadata and workspace_id:
            # Fallback: Update metadata with workspace_id if we got it from authentication
            update_request.metadata.workspace_id = workspace_id
            logger.info(f"Updated update_request.metadata with workspace_id: {update_request.metadata}")

        logger.info(f"Resolved end_user_id: {end_user_id}, developer_id: {developer_id}")
        logger.info(f"is_qwen_route: {is_qwen_route}")
        logger.info(f"workspace_id: {workspace_id}")
        logger.info(f"update_request.metadata: {update_request.metadata}")
        
        # Normalize is_qwen_route: if None or False, default to False
        if is_qwen_route is None:
            is_qwen_route = False
            logger.info(f"is_qwen_route was None, defaulting to False for update memory route")
        
        # legacy_route is the opposite of is_qwen_route
        legacy_route = not is_qwen_route
        logger.info(f"Setting legacy_route for update memory: {legacy_route} (is_qwen_route: {is_qwen_route})")

        # Prepare metadata
        metadata = update_request.metadata.model_dump() if update_request.metadata else {}
        # Do not forcibly overwrite user_id; only set if not present
        if 'user_id' not in metadata or not metadata['user_id']:
            metadata['user_id'] = str(user_id)

        # Patch request.json for legacy handler
        async def get_json():
            return {
                "id": memory_id,
                "content": update_request.content,
                "type": update_request.type,
                "metadata": metadata  # Use the prepared metadata that includes user_id
            }
        request.json = get_json
        await memory_graph.ensure_async_connection()
        async with memory_graph.async_neo_conn.get_session() as neo_session:
            # Update memory item
            result = await memory_graph.update_memory_item(
                session_token=sessionToken, 
                memory_id=memory_id,
                memory_type=update_request.type,
                content=update_request.content, 
                metadata=metadata,
                background_tasks=background_tasks,
                neo_session=neo_session,
                api_key=api_key,
                legacy_route=legacy_route  # Use the normalized legacy_route
            )

            # Process relationships if provided
            if update_request.relationships_json:
                logger.info(f"Processing {len(update_request.relationships_json)} relationships for memory {memory_id}")
                
                # Convert relationships to RelationshipItem objects if they aren't already
                relationship_items = []
                for rel in update_request.relationships_json:
                    if isinstance(rel, dict):
                        # Convert dict to RelationshipItem
                        relationship_items.append(RelationshipItem(**rel))
                    else:
                        # Already a RelationshipItem object
                        relationship_items.append(rel)
                
                # Create a temporary memory item for relationship processing
                # We need to get the existing memory item to process relationships
                existing_memory = await memory_graph.get_memory_item(memory_id, neo_session)
                if existing_memory:
                    # Create a MemoryItem from the existing data for relationship processing
                    from memory.memory_item import TextMemoryItem
                    temp_memory_item = TextMemoryItem(
                        content=existing_memory.get('content', ''),
                        metadata=existing_memory.get('metadata', {}),
                        context=update_request.context or []
                    )
                    temp_memory_item.id = memory_id
                    
                    # Process relationships
                    relationship_result = await memory_graph.update_memory_item_with_relationships(
                        memory_item=temp_memory_item,
                        relationships_json=relationship_items,
                        workspace_id=metadata.get('workspace_id'),
                        user_id=end_user_id,
                        neo_session=neo_session,
                        legacy_route=not is_qwen_route
                    )
                    
                    if relationship_result.get('success'):
                        logger.info(f"Successfully processed relationships for memory {memory_id}")
                    else:
                        logger.warning(f"Failed to process some relationships for memory {memory_id}: {relationship_result.get('error')}")
                else:
                    logger.warning(f"Could not find existing memory {memory_id} for relationship processing")

        # Handle error case
        if hasattr(result, 'code') and getattr(result, 'code', 200) != 200:
            response.status_code = getattr(result, 'code', 500)
            return UpdateMemoryResponse.failure(
                error=getattr(result, 'detail', 'Unknown error'),
                code=getattr(result, 'code', 500)
            )
        elif isinstance(result, dict) and result.get('code', 200) != 200:
            response.status_code = result.get('code', 500)
            return UpdateMemoryResponse.failure(
                error=result.get('detail', 'Unknown error'),
                code=result.get('code', 500)
            )
        elif not result:
            response.status_code = 404
            return UpdateMemoryResponse.failure(
                error="Memory item not found",
                code=404
            )

        # Log telemetry (edition-aware)
        try:
            from core.services.telemetry import get_telemetry
            telemetry = get_telemetry()
            await telemetry.track(
                "update_memory",
                {
                    "client_type": client_type,
                    "memory_id": memory_id,
                    "api_key": api_key,
                },
                user_id=end_user_id,
                developer_id=developer_id
            )
        except Exception as e:
            logger.error(f"Error tracking update_memory: {e}")

        # Return the response in the correct format
        if isinstance(result, UpdateMemoryResponse):
            response.status_code = result.code if hasattr(result, 'code') else 200
            return result
        else:
            # If response is not already an UpdateMemoryResponse, create one
            status_obj = SystemUpdateStatus(
                pinecone=getattr(result, 'pinecone', False),
                neo4j=getattr(result, 'neo4j', False),
                parse=getattr(result, 'parse', True)
            )
            memory_item = UpdateMemoryItem(
                objectId=getattr(result, 'objectId', ''),
                memoryId=memory_id,
                content=update_request.content or '',
                updatedAt=getattr(result, 'updatedAt', datetime.now()),
                memoryChunkIds=getattr(result, 'memoryChunkIds', [])
            )
            response.status_code = 200
            return UpdateMemoryResponse.success(
                memory_items=[memory_item],
                status_obj=status_obj
            )
    except Exception as e:
        logger.error(f"Error in update_memory: {e}", exc_info=True)
        response.status_code = 500
        return UpdateMemoryResponse.failure(
            error=str(e),
            code=500
        )

@router.post("/batch",
    response_model=BatchMemoryResponse,
    responses={
        200: {"model": BatchMemoryResponse, "description": "Memories successfully added"},
        207: {"model": BatchMemoryResponse, "description": "Partial success - some memories failed"},
        400: {"model": BatchMemoryResponse, "description": "Bad request"},
        401: {"model": BatchMemoryResponse, "description": "Unauthorized"},
        403: {"model": BatchMemoryResponse, "description": "Subscription limit reached"},
        413: {"model": BatchMemoryResponse, "description": "Content too large"},
        415: {"model": BatchMemoryResponse, "description": "Unsupported Media Type"},
        500: {"model": BatchMemoryResponse, "description": "Internal server error"}
    },
    description="""Add multiple memory items in a batch with size validation and background processing.
    
    **Authentication Required**:
    One of the following authentication methods must be used:
    - Bearer token in `Authorization` header
    - API Key in `X-API-Key` header
    - Session token in `X-Session-Token` header
    
    **Required Headers**:
    - Content-Type: application/json
    - X-Client-Type: (e.g., 'papr_plugin', 'browser_extension')
    
    The API validates individual memory content size against MAX_CONTENT_LENGTH environment variable (defaults to 15000 bytes).
    """,
    openapi_extra={
        "operationId": "add_memory_batch",
        "x-openai-isConsequential": False
    })
async def add_memory_batch_v1(
    request: Request,
    batch_request: BatchMemoryRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Security(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Security(session_token_header),
    skip_background_processing: bool = Query(False, description="If True, skips adding background tasks for processing"),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
) -> BatchMemoryResponse:
    logger.info(f"=== add_memory_batch_v1 called ===")
    logger.info(f"Request path: {request.url.path}")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Batch request: {batch_request}")
    
    # DEBUG: Check memory_graph MongoDB client at entry point
    logger.info(f"🔍 ENTRY POINT DEBUG: memory_graph type: {type(memory_graph)}")
    logger.info(f"🔍 ENTRY POINT DEBUG: memory_graph.mongo_client: {memory_graph.mongo_client is not None if memory_graph else 'memory_graph is None'}")
    if memory_graph and memory_graph.mongo_client:
        logger.info(f"🔍 ENTRY POINT DEBUG: mongo_client.address: {memory_graph.mongo_client.address}")
    logger.info(f"🔍 ENTRY POINT DEBUG: memory_graph.db: {memory_graph.db.name if memory_graph and memory_graph.db else 'None'}")
    
    """
    Add multiple memory items in a batch with size validation and background processing.
    
    Args:
        request (Request): FastAPI request object
        batch_request (BatchMemoryRequest): Batch of memory items to add
        response (Response): FastAPI response object
        background_tasks (BackgroundTasks): FastAPI background tasks handler
        api_key (Optional[str]): Optional API key for authentication
        bearer_token (Optional[HTTPAuthorizationCredentials]): Optional bearer token for authentication
        skip_background_processing (bool): If True, skips adding background tasks for processing
        memory_graph (MemoryGraph): Per-request MemoryGraph instance
        
    Returns:
        BatchMemoryResponse: Unified envelope containing successful and error results.
    """
    try:
        # Log the incoming request for debugging
        logger.info(f"=== Batch Memory Request Debug ===")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Batch request: {batch_request}")
        logger.info(f"Batch request memories count: {len(batch_request.memories) if batch_request.memories else 0}")
        
        # Handle API Key authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header and api_key:
            # Patch the request headers to include Authorization for downstream code
            request.headers.__dict__['_list'].append(
                (b'authorization', f'APIKey {api_key}'.encode())
            )
            auth_header = request.headers.get('Authorization')

        if not auth_header and not api_key:
            response.status_code = 401
            return BatchMemoryResponse.failure(
                errors=[BatchMemoryError(index=-1, error="Missing authentication")],
                code=401,
                error="Missing authentication"
            )

        # --- Optimized authentication using cached method ---
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')
        auth_start_time = time.time()
        try:
            async with httpx.AsyncClient() as httpx_client:
                # Use optimized authentication that gets workspace_id, isQwenRoute, user_roles, user_workspace_ids, and resolves user in parallel
                if api_key and bearer_token:
                    # Developer provides API key + Bearer token for end user - use Bearer token for auth but developer's API key for Parse
                    auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", client_type, memory_graph, api_key=api_key, search_request=None, memory_request=None, batch_request=batch_request, httpx_client=httpx_client)
                elif api_key and session_token:
                    auth_response = await get_user_from_token_optimized(f"Session {session_token}", client_type, memory_graph, api_key=api_key, search_request=None, memory_request=None, batch_request=batch_request, httpx_client=httpx_client)
                elif api_key:
                    auth_response = await get_user_from_token_optimized(f"APIKey {api_key}", client_type, memory_graph, api_key=api_key, search_request=None, memory_request=None, batch_request=batch_request, httpx_client=httpx_client)
                elif bearer_token:
                    auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", client_type, memory_graph, api_key=api_key, search_request=None, memory_request=None, batch_request=batch_request, httpx_client=httpx_client)
                elif session_token:
                    auth_response = await get_user_from_token_optimized(f"Session {session_token}", client_type, memory_graph, api_key=api_key, search_request=None, memory_request=None, batch_request=batch_request, httpx_client=httpx_client)
                else:
                    auth_response = await get_user_from_token_optimized(auth_header, client_type, memory_graph, api_key=api_key, search_request=None, memory_request=None, batch_request=batch_request, httpx_client=httpx_client)
        except ValueError as e:
            logger.error(f"Invalid authentication token: {e}")
            response.status_code = 401
            return BatchMemoryResponse.failure(
                errors=[BatchMemoryError(index=-1, error="Invalid authentication token")],
                code=401,
                error="Invalid authentication token"
            )
        auth_end_time = time.time()
        logger.info(f"Enhanced authentication timing: {(auth_end_time - auth_start_time) * 1000:.2f}ms")

        # Extract values from the optimized auth response
        user_id = auth_response.developer_id
        logger.info(f"user_id: {user_id}")
        end_user_id = auth_response.end_user_id
        logger.info(f"end_user_id: {end_user_id}")
        sessionToken = auth_response.session_token
        user_info = auth_response.user_info
        api_key = auth_response.api_key
        workspace_id = auth_response.workspace_id
        is_qwen_route = auth_response.is_qwen_route
        user_roles = auth_response.user_roles
        user_workspace_ids = auth_response.user_workspace_ids
        updated_metadata = auth_response.updated_metadata

        # Extract multi-tenant context using utility function
        from services.multi_tenant_utils import extract_multi_tenant_context, apply_multi_tenant_scoping_to_batch_request
        auth_context = extract_multi_tenant_context(auth_response)

        # Set developer_id from user_info
        developer_id = user_info.get("developer_id") if user_info and "developer_id" in user_info else user_id
        logger.info(f"developer_id: {developer_id}")

        # Check interaction limits (4 mini interactions per memory in batch)
        from models.operation_types import MemoryOperationType
        from config.features import get_features
        features = get_features()
        
        batch_size = len(batch_request.memories) if batch_request.memories else 0
        logger.info(f"Batch size: {batch_size}")
        
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
            
            user_instance = User(id=user_id)
            
            # DEBUG: Check memory_graph MongoDB client before limits check
            logger.info(f"🔍 BEFORE LIMITS CHECK DEBUG: memory_graph.mongo_client: {memory_graph.mongo_client is not None if memory_graph else 'memory_graph is None'}")
            logger.info(f"🔍 BEFORE LIMITS CHECK DEBUG: memory_graph.db: {memory_graph.db.name if memory_graph and memory_graph.db else 'None'}")
            
            limit_check = await user_instance.check_interaction_limits_fast(
                interaction_type='mini',
                memory_graph=memory_graph,
                operation=MemoryOperationType.ADD_MEMORY_BATCH,
                batch_size=batch_size,
                api_key_id=api_key_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            if limit_check:
                response_dict, status_code, is_error = limit_check
                if is_error:
                    response.status_code = status_code
                    return BatchMemoryResponse.failure(
                        errors=[BatchMemoryError(index=-1, error=response_dict.get('error'))],
                        code=status_code,
                        error=response_dict.get('error'),
                        message=response_dict.get('message'),
                        details=response_dict
                    )

        # For batch requests, the optimized auth now handles user resolution via patch_and_resolve_user_ids_and_acls
        # The auth_response.end_user_id should already contain the resolved end_user_id
        # We can use the updated_metadata from auth_response if available
        if updated_metadata:
            # The optimized auth has already processed the batch request and resolved user IDs
            # We can use the resolved end_user_id from auth_response
            end_user_id = auth_response.end_user_id
            logger.info(f"Using resolved end_user_id from optimized auth: {end_user_id}")
        else:
            # Fallback: use the end_user_id from auth_response
            end_user_id = auth_response.end_user_id
            logger.info(f"Using end_user_id from auth_response: {end_user_id}")
        # temp till we have SDK support for webhook url
        #if not batch_request.webhook_url:
        #    batch_request.webhook_url = "http://localhost:8000/webhook"
        logger.info(f"Resolved end_user_id: {end_user_id}, developer_id: {developer_id}")
        logger.info(f"is_qwen_route: {is_qwen_route}")
        logger.info(f"workspace_id: {workspace_id}")
        logger.info(f"batch_request: {batch_request}")
        logger.info(f"Webhook URL in batch_request: {getattr(batch_request, 'webhook_url', None)}")
        logger.info(f"Webhook secret in batch_request: {getattr(batch_request, 'webhook_secret', None)}")

        # Update the batch request with resolved user IDs if available from optimized auth
        if auth_response.updated_batch_request:
            # Use the updated batch request from optimized auth
            batch_request = auth_response.updated_batch_request
            logger.info(f"Using updated batch request from optimized auth - end_user_id: {end_user_id}")
        elif updated_metadata:
            # The optimized auth has already processed the batch request
            # We should use the updated batch request from auth_response if available
            # For now, we'll use the original batch_request but with the resolved end_user_id
            logger.info(f"Using optimized auth results - end_user_id: {end_user_id}")
        else:
            # Fallback: ensure batch_request has the resolved user IDs
            batch_request.user_id = end_user_id
            batch_request.external_user_id = None  # Clear external_user_id since we resolved it
            logger.info(f"Updated batch_request with resolved end_user_id: {end_user_id}")

        # Apply multi-tenant scoping using utility function
        apply_multi_tenant_scoping_to_batch_request(batch_request, auth_context)

        # --- TEMPORAL BATCH PROCESSING DECISION ---
        # Check if we should use Temporal for this batch (cloud-only feature)
        from services.batch_processor import should_use_temporal, validate_batch_size, process_batch_with_temporal

        batch_size = len(batch_request.memories)

        # Validate batch size first
        is_valid, error_message, max_allowed = await validate_batch_size(batch_size)
        if not is_valid:
            logger.warning(f"Batch size {batch_size} exceeds limit {max_allowed}")
            response.status_code = 413
            return BatchMemoryResponse.failure(
                errors=[BatchMemoryError(index=-1, error=error_message)],
                code=413,
                error=f"Batch size exceeds limit ({max_allowed})"
            )

        # Check if we should use Temporal for this batch
        use_temporal = await should_use_temporal(batch_size)

        if use_temporal:
            logger.info(f"Using Temporal for batch of {batch_size} memories (cloud edition)")

            # Extract webhook information if available
            webhook_url = getattr(batch_request, 'webhook_url', None)
            webhook_secret = getattr(batch_request, 'webhook_secret', None)

            # Process with Temporal using real authentication data (returns immediately with workflow ID)
            result = await process_batch_with_temporal(
                batch_request=batch_request,
                auth_response=auth_response,
                api_key=api_key,
                webhook_url=webhook_url,
                webhook_secret=webhook_secret
            )

            logger.info(f"Temporal batch processing initiated: {result.details.get('workflow_id') if result.details else 'unknown'}")
            response.status_code = result.code
            return result
        else:
            logger.info(f"Using background tasks for batch of {batch_size} memories (standard processing)")

        # Convert memories to list format expected by common_add_memory_batch_handler
        memory_list = []
        for memory in batch_request.memories:
            # Create a modified request for each memory item
            modified_scope = dict(request.scope)
            # Ensure we copy all headers including Authorization
            headers = []
            for k, v in request.headers.items():
                if k.lower() == 'authorization' and not v:
                    # If Authorization header is empty but we have API key, use that
                    if api_key:
                        headers.append((b'authorization', f'APIKey {api_key}'.encode()))
                else:
                    headers.append((k.lower().encode(), v.encode()))
            modified_scope['headers'] = headers
            modified_request = Request(scope=modified_scope)
            
            # Set the JSON body for this memory item
            memory_dict = memory.as_handler_dict()
            async def get_json():
                return memory_dict
            modified_request.json = get_json
            
            memory_list.append(memory_dict)

        # Note: We're using the original request which has proper auth middleware
        logger.info(f"Using original request with auth middleware")

        # --- Neo4j session management: create one session for the whole batch ---
        
        result = await common_add_memory_batch_handler(
            request=request,  # Use original request with proper auth middleware
            memory_graph=memory_graph,  # Use per-request instance
            background_tasks=background_tasks,
            memory_request_batch=batch_request,
            auth_response=auth_response,
            batch_size=batch_request.batch_size,
            skip_background_processing=skip_background_processing,
            legacy_route=True # this only impacts memory grouping since new memories will automatically go to new qwen route and not the legacy route
        )

        # Log telemetry (edition-aware)
        try:
            from core.services.telemetry import get_telemetry
            telemetry = get_telemetry()
            await telemetry.track(
                "add_memory_batch",
                {
                    "client_type": client_type,
                    "batch_size": batch_request.batch_size,
                    "total_processed": getattr(result, 'total_processed', None),
                    "total_successful": getattr(result, 'total_successful', None),
                    "total_failed": getattr(result, 'total_failed', None),
                    "code": getattr(result, 'code', None),
                    "api_key": api_key,
                },
                user_id=end_user_id,
                developer_id=developer_id
            )
        except Exception as e:
            logger.error(f"Error tracking add_memory_batch: {e}")
        
        # Set status code based on result
        # Always set status code to match result.code
        response.status_code = result.code
        return result
        
    except ValidationError as ve:
        logger.error(f"Validation error in batch request: {ve}")
        logger.error(f"Validation details: {ve.errors()}")
        response.status_code = 422
        return BatchMemoryResponse.failure(
            errors=[BatchMemoryError(
                code=422,
                error=f"Validation error: {ve}",
                index=-1
            )],
            code=422,
            error=f"Validation error: {ve}"
        )
    except Exception as e:
        logger.error(f"Error processing memory batch: {e}", exc_info=True)
        response.status_code = 500
        return BatchMemoryResponse.failure(
            errors=[BatchMemoryError(
                code=500,
                error=str(e),
                index=-1
            )],
            code=500,
            error=str(e)
        )

@router.delete("/all",
    response_model=BatchMemoryResponse,
    responses={
        200: {"model": BatchMemoryResponse, "description": "All memories successfully deleted"},
        207: {"model": BatchMemoryResponse, "description": "Partial success - some memories failed to delete"},
        400: {"model": BatchMemoryResponse, "description": "Bad request"},
        401: {"model": BatchMemoryResponse, "description": "Unauthorized"},
        404: {"model": BatchMemoryResponse, "description": "No memories found for user"},
        500: {"model": BatchMemoryResponse, "description": "Internal server error"}
    },
    description="""Delete all memory items for a user.
    
    **Authentication Required**:
    One of the following authentication methods must be used:
    - Bearer token in `Authorization` header
    - API Key in `X-API-Key` header
    - Session token in `X-Session-Token` header
    
    **User Resolution**:
    - If only API key is provided: deletes memories for the developer
    - If user_id or external_user_id is provided: resolves and deletes memories for that user
    - Uses the same user resolution logic as other endpoints
    
    **Required Headers**:
    - X-Client-Type: (e.g., 'papr_plugin', 'browser_extension')
    
    **WARNING**: This operation cannot be undone. All memories for the resolved user will be permanently deleted.
    """,
    openapi_extra={
        "operationId": "delete_all_memories",
        "x-openai-isConsequential": True
    }
)
async def delete_all_memories_v1(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Security(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Security(session_token_header),
    user_id: Optional[str] = Query(None, description="Optional user ID to delete memories for (if not provided, uses authenticated user)"),
    external_user_id: Optional[str] = Query(None, description="Optional external user ID to resolve and delete memories for"),
    skip_parse: bool = Query(False, description="Skip Parse Server deletion"),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
) -> BatchMemoryResponse:
    """
    Delete all memory items for a user.
    """
    try:
        logger.info(f"[delete_all_memories_v1] Received request - user_id: {user_id}, external_user_id: {external_user_id}")
        
        # Get client type
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')

        # Preserve original request parameters before we overwrite local variables below
        original_user_id = user_id
        original_external_user_id = external_user_id

        # Authenticate user using optimized authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header and api_key:
            request.headers.__dict__['_list'].append(
                (b'authorization', f'APIKey {api_key}'.encode())
            )
            auth_header = request.headers.get('Authorization')

        if not auth_header and not api_key and not session_token:
            return BatchMemoryResponse.failure(
                errors=[BatchMemoryError(index=-1, error="Missing authentication")],
                code=401,
                error="Missing authentication"
            )

        # Create a mock search request for user resolution if user_id/external_user_id provided
        mock_search_request = None
        if user_id or external_user_id:
            from models.memory_models import SearchRequest, MemoryMetadata
            mock_metadata = MemoryMetadata()
            if user_id:
                mock_metadata.user_id = user_id
            if external_user_id:
                mock_metadata.external_user_id = external_user_id
            mock_search_request = SearchRequest(
                query="mock", # Required but not used for user resolution
                metadata=mock_metadata
            )

        # --- Optimized authentication using cached method ---
        auth_start_time = time.time()
        try:
            async with httpx.AsyncClient() as httpx_client:
                if api_key and bearer_token:
                    auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", client_type, memory_graph, api_key=api_key, search_request=mock_search_request, httpx_client=httpx_client)
                elif api_key and session_token:
                    auth_response = await get_user_from_token_optimized(f"Session {session_token}", client_type, memory_graph, api_key=api_key, search_request=mock_search_request, httpx_client=httpx_client)
                elif api_key:
                    auth_response = await get_user_from_token_optimized(f"APIKey {api_key}", client_type, memory_graph, api_key=api_key, search_request=mock_search_request, httpx_client=httpx_client)
                elif bearer_token:
                    auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", client_type, memory_graph, search_request=mock_search_request, httpx_client=httpx_client)
                elif session_token:
                    auth_response = await get_user_from_token_optimized(f"Session {session_token}", client_type, memory_graph, search_request=mock_search_request, httpx_client=httpx_client)
                else:
                    auth_response = await get_user_from_token_optimized(auth_header, client_type, memory_graph, search_request=mock_search_request, httpx_client=httpx_client)
        except ValueError as e:
            logger.error(f"Invalid authentication token: {e}")
            return BatchMemoryResponse.failure(
                errors=[BatchMemoryError(index=-1, error="Invalid authentication token")],
                code=401,
                error="Invalid authentication token"
            )
        auth_end_time = time.time()
        logger.info(f"Enhanced authentication timing: {(auth_end_time - auth_start_time) * 1000:.2f}ms")

        # Extract values from the optimized auth response
        user_id = auth_response.developer_id
        end_user_id = auth_response.end_user_id
        sessionToken = auth_response.session_token
        user_info = auth_response.user_info
        api_key = auth_response.api_key
        workspace_id = auth_response.workspace_id
        is_qwen_route = auth_response.is_qwen_route
        user_roles = auth_response.user_roles
        user_workspace_ids = auth_response.user_workspace_ids
        updated_metadata = auth_response.updated_metadata
        
        # Set developer_id from user_info
        developer_id = user_info.get("developer_id") if user_info and "developer_id" in user_info else user_id
        logger.info(f"developer_id: {developer_id}")
        
        # Check interaction limits (1 mini interaction for delete_all_memories)
        from models.operation_types import MemoryOperationType
        from config.features import get_features
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
            
            user_instance = User(id=user_id)
            limit_check = await user_instance.check_interaction_limits_fast(
                interaction_type='mini',
                memory_graph=memory_graph,
                operation=MemoryOperationType.DELETE_ALL_MEMORIES,
                api_key_id=api_key_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            if limit_check:
                response_dict, status_code, is_error = limit_check
                if is_error:
                    response.status_code = status_code
                    return BatchMemoryResponse.failure(
                        errors=[BatchMemoryError(index=-1, error=response_dict.get('error'))],
                        code=status_code,
                        error=response_dict.get('error'),
                        message=response_dict.get('message'),
                        details=response_dict
                    )
        
        # For delete all, the optimized auth now handles user resolution via patch_and_resolve_user_ids_and_acls
        # The auth_response.end_user_id should already contain the resolved end_user_id
        # We can use the updated_metadata from auth_response if available
        if updated_metadata:
            # The optimized auth has already processed the request and resolved user IDs
            # We can use the resolved end_user_id from auth_response
            end_user_id = auth_response.end_user_id
            logger.info(f"Using resolved end_user_id from optimized auth: {end_user_id}")
        else:
            # Fallback: use the end_user_id from auth_response
            end_user_id = auth_response.end_user_id
            logger.info(f"Using end_user_id from auth_response: {end_user_id}")
        
        # Resolve the user ID to delete memories for using the ORIGINAL request parameters (from query params):
        # 1) If user_id provided in query, use it as-is (do not overwrite with developer_id)
        # 2) Else if external_user_id provided, resolve it to internal user_id
        # 3) Else fall back to the authenticated end_user_id
        provided_user_id = request.query_params.get('user_id')
        provided_external_user_id = request.query_params.get('external_user_id')
        resolved_user_id = None
        if provided_user_id:
            resolved_user_id = provided_user_id
            logger.info(f"Using provided user_id: {resolved_user_id}")
        elif provided_external_user_id:
            # Resolve external_user_id to internal user_id using User service
            from models.memory_models import MemoryMetadata
            temp_metadata = MemoryMetadata(external_user_id=provided_external_user_id)
            async with httpx.AsyncClient() as httpx_client:
                user_service = User(user_id)
                updated_metadata, id_map, _ = await user_service.resolve_external_user_ids_to_internal(
                    developer_id=user_id,
                    metadata=temp_metadata,
                    httpx_client=httpx_client,
                    x_api_key=api_key
                )
                resolved_user_id = getattr(updated_metadata, 'user_id', None)
            logger.info(f"Resolved external_user_id {provided_external_user_id} to internal user_id: {resolved_user_id}")
        else:
            resolved_user_id = end_user_id
            logger.info(f"Using authenticated user_id: {resolved_user_id}")

        # Guardrails: If caller provided user_id/external_user_id, do NOT allow deletion to target the developer's own user by mistake
        # Allow only if the caller explicitly passed the developer_id as user_id (intentional), otherwise block
        if (provided_user_id or provided_external_user_id) and resolved_user_id == user_id and (not provided_user_id or provided_user_id != user_id):
            logger.warning(
                "Refusing to delete developer's memories when request targeted another user. "
                f"developer_id={user_id}, provided_user_id={provided_user_id}, provided_external_user_id={provided_external_user_id}, resolved_user_id={resolved_user_id}"
            )
            response.status_code = 400
            return BatchMemoryResponse.failure(
                errors=[BatchMemoryError(index=-1, error="Resolved user matched developer's memories for a request targeting an end user")],
                code=400,
                error="Resolved user matched developer's memories for a request targeting an end user"
            )
        
        if not resolved_user_id:
            response.status_code = 400
            return BatchMemoryResponse.failure(
                errors=[BatchMemoryError(index=-1, error="Could not resolve user ID")],
                code=400,
                error="Could not resolve user ID"
            )
        
        logger.info(f"Will delete all memories for user: {resolved_user_id}")
        
        # Get all memories for the user
        try:
            # Use the existing service to get all memory IDs for the user
            from services.memory_management import retrieve_all_memory_ids_for_user
            
            all_memory_ids = await retrieve_all_memory_ids_for_user(
                session_token=sessionToken,
                user_id=resolved_user_id,
                api_key=api_key
            )
            logger.info(f"Found {len(all_memory_ids)} memories to delete for user {resolved_user_id}")
            
            if not all_memory_ids:
                response.status_code = 404
                return BatchMemoryResponse.failure(
                    errors=[BatchMemoryError(index=-1, error="No memories found for user")],
                    code=404,
                    error="No memories found for user"
                )
            
            # Batch delete all memories
            total_successful = 0
            total_failed = 0
            errors = []
            
            # Ensure we have a Neo4j connection
            await memory_graph.ensure_async_connection()
            
            for i, memory_id in enumerate(all_memory_ids):
                try:
                    # Delete the memory using Neo4j session
                    async with memory_graph.async_neo_conn.get_session() as neo_session:
                        delete_result = await memory_graph.delete_memory_item(
                            memory_id=memory_id,
                            session_token=sessionToken,
                            neo_session=neo_session,
                            skip_parse=skip_parse,
                            api_key=api_key,
                            legacy_route=True
                        )
                        
                        if delete_result.success:
                            total_successful += 1
                            logger.info(f"Successfully deleted memory {memory_id}")
                        else:
                            total_failed += 1
                            error_msg = f"Failed to delete memory {memory_id}: {delete_result.error}"
                            errors.append(BatchMemoryError(index=i, error=error_msg))
                            logger.error(error_msg)
                            
                except Exception as e:
                    total_failed += 1
                    error_msg = f"Exception deleting memory {memory_id}: {str(e)}"
                    errors.append(BatchMemoryError(index=i, error=error_msg))
                    logger.error(error_msg, exc_info=True)
            
            # Log telemetry (edition-aware)
            try:
                from core.services.telemetry import get_telemetry
                telemetry = get_telemetry()
                await telemetry.track(
                    "delete_all_memories",
                    {
                        "client_type": client_type,
                        "total_memories": len(all_memory_ids),
                        "total_successful": total_successful,
                        "total_failed": total_failed,
                        "api_key": api_key,
                    },
                    user_id=end_user_id,
                    developer_id=developer_id
                )
            except Exception as e:
                logger.error(f"Error tracking delete_all_memories: {e}")
            
            # Set status code based on results
            if total_failed == 0:
                response.status_code = 200
                return BatchMemoryResponse.success(
                    successful=[],
                    total_processed=len(all_memory_ids),
                    total_successful=total_successful,
                    total_failed=total_failed,
                    details={"message": f"Successfully deleted all {total_successful} memories for user {resolved_user_id}"}
                )
            elif total_successful > 0:
                response.status_code = 207
                return BatchMemoryResponse.partial_success(
                    successful=[],
                    errors=errors,
                    total_processed=len(all_memory_ids),
                    total_successful=total_successful,
                    total_failed=total_failed,
                    details={"message": f"Partially successful - deleted {total_successful} memories, failed to delete {total_failed} memories"}
                )
            else:
                response.status_code = 500
                return BatchMemoryResponse.failure(
                    errors=errors,
                    code=500,
                    error=f"Failed to delete all {total_failed} memories",
                    details={"message": f"Failed to delete all {total_failed} memories"}
                )
                
        except Exception as e:
            logger.error(f"Error getting memories for user {resolved_user_id}: {e}", exc_info=True)
            response.status_code = 500
            return BatchMemoryResponse.failure(
                errors=[BatchMemoryError(index=-1, error=f"Error getting memories: {str(e)}")],
                code=500,
                error=f"Error getting memories: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"Error in delete_all_memories_v1: {e}", exc_info=True)
        response.status_code = 500
        return BatchMemoryResponse.failure(
            errors=[BatchMemoryError(index=-1, error=str(e))],
            code=500,
            error=str(e)
        )

@router.delete("/{memory_id}",
    response_model=DeleteMemoryResponse,
    responses={
        200: {"model": DeleteMemoryResponse, "description": "Memory successfully deleted"},
        207: {"model": DeleteMemoryResponse, "description": "Partially successful deletion"},
        400: {"model": DeleteMemoryResponse, "description": "Bad request"},
        401: {"model": DeleteMemoryResponse, "description": "Unauthorized"},
        404: {"model": DeleteMemoryResponse, "description": "Memory not found"},
        500: {"model": DeleteMemoryResponse, "description": "Internal server error"}
    },
    description="""Delete a memory item by ID.
    
    **Authentication Required**:
    One of the following authentication methods must be used:
    - Bearer token in `Authorization` header
    - API Key in `X-API-Key` header
    - Session token in `X-Session-Token` header
    
    **Required Headers**:
    - X-Client-Type: (e.g., 'papr_plugin', 'browser_extension')
    """,
    openapi_extra={
        "operationId": "delete_memory",
        "x-openai-isConsequential": True
    }
    )
async def delete_memory_v1(
    memory_id: str,
    request: Request,
    response: Response,
    api_key: Optional[str] = Security(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Security(session_token_header),
    skip_parse: bool = Query(False, description="Skip Parse Server deletion"),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
) -> DeleteMemoryResponse:
    """
    Delete a memory item by ID.
    """
    try:
        logger.info(f"[delete_memory_v1] Received request to delete memory_id: {memory_id}")
        # Get client type
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')

        # Authenticate user using optimized authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header and api_key:
            # Patch the request headers to include Authorization for downstream code
            request.headers.__dict__['_list'].append(
                (b'authorization', f'APIKey {api_key}'.encode())
            )
            auth_header = request.headers.get('Authorization')

        if not auth_header and not api_key and not session_token:
            response.status_code = 401
            return DeleteMemoryResponse.failure(
                error="Missing authentication",
                code=401
            )

        # --- Optimized authentication using cached method ---
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')
        auth_start_time = time.time()
        try:
            async with httpx.AsyncClient() as httpx_client:
                # Use optimized authentication that gets workspace_id, isQwenRoute, user_roles, user_workspace_ids, and resolves user in parallel
                if api_key and bearer_token:
                    # Developer provides API key + Bearer token for end user - use Bearer token for auth but developer's API key for Parse
                    auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", client_type, memory_graph, api_key=api_key, httpx_client=httpx_client)
                elif api_key and session_token:
                    auth_response = await get_user_from_token_optimized(f"Session {session_token}", client_type, memory_graph, api_key=api_key, httpx_client=httpx_client)
                elif api_key:
                    auth_response = await get_user_from_token_optimized(f"APIKey {api_key}", client_type, memory_graph, httpx_client=httpx_client)
                elif bearer_token:
                    auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", client_type, memory_graph, httpx_client=httpx_client)
                elif session_token:
                    auth_response = await get_user_from_token_optimized(f"Session {session_token}", client_type, memory_graph, httpx_client=httpx_client)
                else:
                    auth_response = await get_user_from_token_optimized(auth_header, client_type, memory_graph, httpx_client=httpx_client)
        except ValueError as e:
            logger.error(f"Invalid authentication token: {e}")
            response.status_code = 401
            return DeleteMemoryResponse.failure(
                error="Invalid authentication token",
                code=401
            )
        auth_end_time = time.time()
        logger.info(f"Enhanced authentication timing: {(auth_end_time - auth_start_time) * 1000:.2f}ms")

        # Extract values from the optimized auth response
        user_id = auth_response.developer_id
        end_user_id = auth_response.end_user_id
        sessionToken = auth_response.session_token
        user_info = auth_response.user_info
        api_key = auth_response.api_key
        workspace_id = auth_response.workspace_id
        is_qwen_route = auth_response.is_qwen_route
        user_roles = auth_response.user_roles
        user_workspace_ids = auth_response.user_workspace_ids
        updated_metadata = auth_response.updated_metadata
        
        # Set developer_id from user_info
        developer_id = user_info.get("developer_id") if user_info and "developer_id" in user_info else user_id
        
        # Check interaction limits (1 mini interaction for delete_memory)
        from models.operation_types import MemoryOperationType
        from config.features import get_features
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
            
            user_instance = User(id=user_id)
            limit_check = await user_instance.check_interaction_limits_fast(
                interaction_type='mini',
                memory_graph=memory_graph,
                operation=MemoryOperationType.DELETE_MEMORY,
                api_key_id=api_key_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            if limit_check:
                response_dict, status_code, is_error = limit_check
                if is_error:
                    response.status_code = status_code
                    return DeleteMemoryResponse.failure(
                        error=response_dict.get('error'),
                        code=status_code,
                        message=response_dict.get('message'),
                        details=response_dict
                    )
        
        # Determine legacy route based on is_qwen_route flag
        legacy_route = not is_qwen_route if is_qwen_route is not None else False
        logger.info(f"Using isQwenRoute from optimized auth: {is_qwen_route}")
        logger.info(f"Setting legacy_route for delete: {legacy_route}")

        await memory_graph.ensure_async_connection()
        async with memory_graph.async_neo_conn.get_session() as neo_session:
            logger.info(f"[delete_memory_v1] Calling delete_memory_item with memory_id: {memory_id}, skip_parse: {skip_parse}, api_key: {api_key}, legacy_route: {legacy_route}")
            # Delete memory with legacy_route flag
            result = await memory_graph.delete_memory_item(
                memory_id, 
                sessionToken, 
                neo_session, 
                skip_parse, 
                api_key=api_key,
                legacy_route=legacy_route
            )
        
        response.status_code = result.code if result.code else 200

        # If skip_parse is True, we should mark Parse deletion as successful
        if skip_parse and result.deletion_status:
            result.deletion_status.parse = True

        # Log telemetry (edition-aware)
        try:
            from core.services.telemetry import get_telemetry
            telemetry = get_telemetry()
            await telemetry.track(
                "delete_memory",
                {
                    "client_type": client_type,
                    "memoryId": result.memoryId,
                    "objectId": result.objectId,
                    "deletion_status": result.deletion_status.model_dump() if result.deletion_status else {},
                    "code": result.code,
                    "is_qwen_route": is_qwen_route,
                    "legacy_route": legacy_route,
                    "api_key": api_key,
                },
                user_id=end_user_id,
                developer_id=developer_id
            )
        except Exception as e:
            logger.error(f"Error tracking delete_memory: {e}")

        return result

    except Exception as e:
        logger.error(f"Error in delete_memory_v1: {e}", exc_info=True)
        response.status_code = 500
        return DeleteMemoryResponse.failure(
            error=str(e),
            code=500
        )

@router.get("/{memory_id}",
    response_model=SearchResponse,
    responses={
        200: {"model": SearchResponse, "description": "Memory successfully retrieved"},
        400: {"model": SearchResponse, "description": "Bad request"},
        401: {"model": SearchResponse, "description": "Unauthorized"},
        404: {"model": SearchResponse, "description": "Memory not found"},
        500: {"model": SearchResponse, "description": "Internal server error"}
    },
    description="""Retrieve a memory item by ID.
    
    **Authentication Required**:
    One of the following authentication methods must be used:
    - Bearer token in `Authorization` header
    - API Key in `X-API-Key` header
    - Session token in `X-Session-Token` header
    
    **Required Headers**:
    - X-Client-Type: (e.g., 'papr_plugin', 'browser_extension')
    """,
    openapi_extra={
        "operationId": "get_memory",
        "x-openai-isConsequential": False
    })
async def get_memory_v1(
    memory_id: str,
    request: Request,
    response: Response,
    api_key: Optional[str] = Security(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Security(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph),
    # OMO Safety Filtering Parameters
    require_consent: bool = Query(
        False,
        description="If true, return 404 if the memory has consent='none'. Ensures only consented memories are returned."
    ),
    exclude_flagged: bool = Query(
        False,
        description="If true, return 404 if the memory has risk='flagged'. Filters out flagged content."
    ),
    max_risk: Optional[str] = Query(
        None,
        description="Maximum risk level allowed. Values: 'none', 'sensitive', 'flagged'. If memory exceeds this, return 404."
    )
) -> SearchResponse:
    """
    Retrieve a memory item by ID.

    Supports OMO safety filtering via query parameters:
    - require_consent: Only return memories with consent != 'none'
    - exclude_flagged: Exclude memories with risk='flagged'
    - max_risk: Set maximum allowed risk level ('none', 'sensitive', 'flagged')
    """
    try:
        # Get client type
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')

        # Authenticate user
        auth_header = request.headers.get('Authorization')
        if not auth_header and api_key:
            # Patch the request headers to include Authorization for downstream code
            request.headers.__dict__['_list'].append(
                (b'authorization', f'APIKey {api_key}'.encode())
            )
            auth_header = request.headers.get('Authorization')

        if not auth_header and not api_key and not session_token:
            result = SearchResponse.failure(
                error="Missing authentication",
                code=401
            )
            response.status_code = result.code
            return result

        # --- Optimized authentication using cached method ---
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')
        auth_start_time = time.time()
        try:
            async with httpx.AsyncClient() as httpx_client:
                # Use optimized authentication that gets workspace_id, isQwenRoute, user_roles, user_workspace_ids, and resolves user in parallel
                if api_key and bearer_token:
                    # Developer provides API key + Bearer token for end user - use Bearer token for auth but developer's API key for Parse
                    auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", client_type, memory_graph, api_key=api_key, httpx_client=httpx_client)
                elif api_key and session_token:
                    auth_response = await get_user_from_token_optimized(f"Session {session_token}", client_type, memory_graph, api_key=api_key, httpx_client=httpx_client)
                elif api_key:
                    auth_response = await get_user_from_token_optimized(f"APIKey {api_key}", client_type, memory_graph, httpx_client=httpx_client)
                elif bearer_token:
                    auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", client_type, memory_graph, httpx_client=httpx_client)
                elif session_token:
                    auth_response = await get_user_from_token_optimized(f"Session {session_token}", client_type, memory_graph, httpx_client=httpx_client)
                else:
                    auth_response = await get_user_from_token_optimized(auth_header, client_type, memory_graph, httpx_client=httpx_client)
        except ValueError as e:
            logger.error(f"Invalid authentication token: {e}")
            result = SearchResponse.failure(
                error="Invalid authentication token",
                code=401
            )
            response.status_code = result.code
            return result
        auth_end_time = time.time()
        logger.info(f"Enhanced authentication timing: {(auth_end_time - auth_start_time) * 1000:.2f}ms")

        # Extract values from the optimized auth response
        user_id = auth_response.developer_id
        end_user_id = auth_response.end_user_id
        session_token = auth_response.session_token
        user_info = auth_response.user_info
        api_key = auth_response.api_key
        workspace_id = auth_response.workspace_id
        is_qwen_route = auth_response.is_qwen_route
        user_roles = auth_response.user_roles
        user_workspace_ids = auth_response.user_workspace_ids
        updated_metadata = auth_response.updated_metadata
        
        # Set developer_id from user_info
        developer_id = user_info.get("developer_id") if user_info and "developer_id" in user_info else user_id
        
        # Check interaction limits (1 mini interaction for get_memory)
        from models.operation_types import MemoryOperationType
        from config.features import get_features
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
            
            user_instance = User(id=user_id)
            limit_check = await user_instance.check_interaction_limits_fast(
                interaction_type='mini',
                memory_graph=memory_graph,
                operation=MemoryOperationType.GET_MEMORY,
                api_key_id=api_key_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            if limit_check:
                response_dict, status_code, is_error = limit_check
                if is_error:
                    result = SearchResponse.failure(
                        error=response_dict.get('error'),
                        code=status_code,
                        message=response_dict.get('message')
                    )
                    response.status_code = result.code
                    return result
        
        logger.info(f"Resolved end_user_id: {end_user_id}, developer_id: {developer_id}")
        logger.info(f"is_qwen_route: {is_qwen_route}")
        logger.info(f"workspace_id: {workspace_id}")

        # --- Use retrieve_memory_item_by_qdrant_id ---
        parse_data = await retrieve_memory_item_by_qdrant_id(session_token, memory_id, api_key=api_key)
        if not parse_data:
            result = SearchResponse.failure(
                error="Memory item not found",
                code=404
            )
            response.status_code = result.code
            return result

        # Convert dict to ParseStoredMemory
        memory_item = ParseStoredMemory.from_dict(parse_data)
        # Convert to public Memory model
        memory = Memory.from_internal(memory_item)

        # OMO Safety Filtering - check consent and risk levels
        if require_consent or exclude_flagged or max_risk:
            # Extract OMO fields from metadata
            metadata = parse_data.get('metadata', {}) or {}
            memory_consent = metadata.get('consent', 'implicit')
            memory_risk = metadata.get('risk', 'none')

            # Check require_consent filter
            if require_consent and memory_consent == 'none':
                logger.info(f"OMO filter: Memory {memory_id} blocked - consent='none' but require_consent=True")
                result = SearchResponse.failure(
                    error="Memory not accessible - missing consent",
                    code=404,
                    message="This memory does not have recorded consent and require_consent filter is enabled."
                )
                response.status_code = result.code
                return result

            # Check exclude_flagged filter
            if exclude_flagged and memory_risk == 'flagged':
                logger.info(f"OMO filter: Memory {memory_id} blocked - risk='flagged' and exclude_flagged=True")
                result = SearchResponse.failure(
                    error="Memory not accessible - flagged content",
                    code=404,
                    message="This memory contains flagged content and exclude_flagged filter is enabled."
                )
                response.status_code = result.code
                return result

            # Check max_risk filter
            if max_risk:
                risk_order = {'none': 0, 'sensitive': 1, 'flagged': 2}
                max_risk_level = risk_order.get(max_risk, 2)
                memory_risk_level = risk_order.get(memory_risk, 0)
                if memory_risk_level > max_risk_level:
                    logger.info(f"OMO filter: Memory {memory_id} blocked - risk='{memory_risk}' exceeds max_risk='{max_risk}'")
                    result = SearchResponse.failure(
                        error="Memory not accessible - exceeds risk threshold",
                        code=404,
                        message=f"This memory has risk level '{memory_risk}' which exceeds the max_risk='{max_risk}' filter."
                    )
                    response.status_code = result.code
                    return result

        # Build SearchResult
        search_result = SearchResult(memories=[memory], nodes=[])
        # Return as SearchResponse
        
        # Log telemetry (edition-aware)
        try:
            from core.services.telemetry import get_telemetry
            telemetry = get_telemetry()
            await telemetry.track(
                "get_memory",
                {
                    "client_type": client_type,
                    "memory_id": memory_id,
                    "api_key": api_key,
                },
                user_id=end_user_id,
                developer_id=developer_id
            )
        except Exception as e:
            logger.error(f"Error tracking get_memory: {e}")

        # Return the response
        result = SearchResponse.success(
            data=search_result,
            code=200
        )
        response.status_code = result.code
        return result

    except Exception as e:
        logger.error(f"Error in get_memory: {e}", exc_info=True)
        result = SearchResponse.failure(
            error=str(e),
            code=500
        )
        response.status_code = result.code
        return result


@router.post("/search", 
    response_model=SearchResponse,
    responses={
        200: {
            "model": SearchResponse, 
            "description": "Successfully retrieved memories",
            "content": {
                "application/json": {
                    "examples": {
                        "system_nodes_response": {
                            "summary": "Response with system nodes only",
                            "description": "Standard response when only system-defined node types are found",
                            "value": {
                                "code": 200,
                                "status": "success",
                                "data": {
                                    "memories": [
                                        {
                                            "id": "mem-123",
                                            "content": "John Doe completed the quarterly report",
                                            "created_at": "2024-01-15T10:30:00Z"
                                        }
                                    ],
                                    "nodes": [
                                        {
                                            "label": "Person",
                                            "properties": {
                                                "id": "person-123",
                                                "name": "John Doe",
                                                "role": "Manager"
                                            },
                                            "schema_id": None
                                        },
                                        {
                                            "label": "Task",
                                            "properties": {
                                                "id": "task-456",
                                                "title": "Quarterly Report",
                                                "status": "completed"
                                            },
                                            "schema_id": None
                                        }
                                    ],
                                    "schemas_used": None
                                },
                                "search_id": "search-789"
                            }
                        },
                        "custom_schema_response": {
                            "summary": "Response with custom schema nodes",
                            "description": "Response when custom UserGraphSchema nodes are found",
                            "value": {
                                "code": 200,
                                "status": "success",
                                "data": {
                                    "memories": [
                                        {
                                            "id": "mem-456",
                                            "content": "Rachel Green implemented the validateForm() function",
                                            "created_at": "2024-01-15T14:20:00Z"
                                        }
                                    ],
                                    "nodes": [
                                        {
                                            "label": "Developer",
                                            "properties": {
                                                "id": "dev-789",
                                                "name": "Rachel Green",
                                                "expertise": ["React", "TypeScript"],
                                                "years_experience": 5
                                            },
                                            "schema_id": "schema_abc123"
                                        },
                                        {
                                            "label": "Function",
                                            "properties": {
                                                "id": "func-101",
                                                "name": "validateForm",
                                                "language": "JavaScript",
                                                "description": "Form validation function"
                                            },
                                            "schema_id": "schema_abc123"
                                        }
                                    ],
                                    "schemas_used": ["schema_abc123"]
                                },
                                "search_id": "search-101"
                            }
                        },
                        "mixed_nodes_response": {
                            "summary": "Response with both system and custom nodes",
                            "description": "Response containing both system nodes and custom schema nodes",
                            "value": {
                                "code": 200,
                                "status": "success",
                                "data": {
                                    "memories": [
                                        {
                                            "id": "mem-789",
                                            "content": "Product launch meeting with development team",
                                            "created_at": "2024-01-15T16:00:00Z"
                                        }
                                    ],
                                    "nodes": [
                                        {
                                            "label": "Meeting",
                                            "properties": {
                                                "id": "meeting-123",
                                                "title": "Product Launch Meeting",
                                                "date": "2024-01-15T16:00:00Z"
                                            },
                                            "schema_id": None
                                        },
                                        {
                                            "label": "Product",
                                            "properties": {
                                                "id": "prod-456",
                                                "name": "Mobile App v2.0",
                                                "category": "Software",
                                                "price": 29.99
                                            },
                                            "schema_id": "schema_ecommerce_456"
                                        }
                                    ],
                                    "schemas_used": ["schema_ecommerce_456"]
                                },
                                "search_id": "search-202"
                            }
                        }
                    }
                }
            }
        },
        400: {"model": SearchResponse, "description": "Bad request"},
        401: {"model": SearchResponse, "description": "Unauthorized"},
        403: {"model": SearchResponse, "description": "Rate limit exceeded"},
        404: {"model": SearchResponse, "description": "No relevant items found"},
        415: {"model": SearchResponse, "description": "Unsupported Media Type"},
        500: {"model": SearchResponse, "description": "Internal server error"}
    },
    description="""Search through memories with authentication required.
    
    **Authentication Required**:
    One of the following authentication methods must be used:
    - Bearer token in `Authorization` header
    - API Key in `X-API-Key` header
    - Session token in `X-Session-Token` header
    
    **Response Format Options**:
    Choose between standard JSON or TOON (Token-Oriented Object Notation) format:
    - **JSON (default)**: Standard JSON response format
    - **TOON**: Optimized format achieving 30-60% token reduction for LLM contexts
      - Use `response_format=toon` query parameter
      - Returns `text/plain` with TOON-formatted content
      - Ideal for LLM integrations to reduce API costs and latency
      - Maintains semantic clarity while minimizing token usage
      - Example: `/v1/memory/search?response_format=toon`
    
    **Custom Schema Support**:
    This endpoint supports both system-defined and custom user-defined node types:
    - **System nodes**: Memory, Person, Company, Project, Task, Insight, Meeting, Opportunity, Code
    - **Custom nodes**: Defined by developers via UserGraphSchema (e.g., Developer, Product, Customer, Function)
    
    When custom schema nodes are returned:
    - Each custom node includes a `schema_id` field referencing the UserGraphSchema
    - The response includes a `schemas_used` array listing all schema IDs used
    - Use `GET /v1/schemas/{schema_id}` to retrieve full schema definitions including:
      - Node type definitions and properties
      - Relationship type definitions and constraints
      - Validation rules and requirements
    
    **Recommended Headers**:
    ```
    Accept-Encoding: gzip
    ```
    
    The API supports response compression for improved performance. Responses larger than 1KB will be automatically compressed when this header is present.
    
    **HIGHLY RECOMMENDED SETTINGS FOR BEST RESULTS:**
    - Set `enable_agentic_graph: true` for intelligent, context-aware search that can understand ambiguous references
    - Use `max_memories: 15-20` for comprehensive memory coverage
    - Use `max_nodes: 10-15` for comprehensive graph entity relationships
    - Use `response_format: toon` when integrating with LLMs to reduce token costs by 30-60%
    
    **Agentic Graph Benefits:**
    When enabled, the system can understand vague references by first identifying specific entities from your memory graph, then performing targeted searches. For example:
    - "customer feedback" → identifies your customers first, then finds their specific feedback
    - "project issues" → identifies your projects first, then finds related issues
    - "team meeting notes" → identifies your team members first, then finds meeting notes
    - "code functions" → identifies your functions first, then finds related code
    
    **Role-Based Memory Filtering:**
    Filter memories by role and category using metadata fields:
    - `metadata.role`: Filter by "user" or "assistant" 
    - `metadata.category`: Filter by category (user: preference, task, goal, facts, context | assistant: skills, learning)
    
    **User Resolution Precedence:**
    - If both user_id and external_user_id are provided, user_id takes precedence.
    - If only external_user_id is provided, it will be resolved to the internal user.
    - If neither is provided, the authenticated user is used.
    """,
    openapi_extra={
        "operationId": "search_memory",
        "x-openai-isConsequential": False,
        "parameters": [
            {
                "name": "Accept-Encoding",
                "in": "header",
                "description": "Recommended to use 'gzip' for response compression",
                "schema": {
                    "type": "string",
                    "default": "gzip"
                },
                "required": False
            }
        ]
    }
)
async def search_v1(
    request: Request,
    search_request: SearchRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Security(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Security(session_token_header),
    max_memories: int = Query(20, description="HIGHLY RECOMMENDED: Maximum number of memories to return. Use at least 15-20 for comprehensive results. Lower values (5-10) may miss relevant information. Default is 20 for optimal coverage.", ge=10, le=50),
    max_nodes: int = Query(15, description="HIGHLY RECOMMENDED: Maximum number of neo nodes to return. Use at least 10-15 for comprehensive graph results. Lower values may miss important entity relationships. Default is 15 for optimal coverage.", ge=10, le=50),
    response_format: ResponseFormat = Query(ResponseFormat.JSON, description="Response format: 'json' (default) or 'toon' (Token-Oriented Object Notation for 30-60% token reduction in LLM contexts)"),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
) -> Union[SearchResponse, PlainTextResponse]:
    """
    Retrieve relevant memory items based on query and context.
    """
    # Start timing for performance metrics
    search_start_time = time.time()
    retrieval_start_time = None
    
    # --- Generate search_id immediately ---
    search_id = str(uuid.uuid4())
    
    try:
        global client
        
        print(f"🚀 SEARCH ROUTE HIT: query='{search_request.query[:30]}...'")  # Debug at route start
        logger.warning(f"🚀 SEARCH ROUTE HIT: query='{search_request.query[:30]}...'")
        
        logger.info(f"api_key: {api_key}")
        # Initialize variables
        chat_gpt = request.app.state.chat_gpt 
        data: Dict[str, Any] = {}

        # Get enable_agentic_graph from SearchRequest body
        enable_agentic_graph = search_request.enable_agentic_graph
        skip_neo = not enable_agentic_graph
        logger.info(f"🔍 AGENTIC CONFIG: enable_agentic_graph={enable_agentic_graph}, skip_neo={skip_neo}")

        # Get content type and validate
        content_type = request.headers.get('Content-Type')
        logger.info(f"Received content_type from client: {content_type}")

        if content_type == 'application/json':
            data = await request.json()
        elif content_type == 'application/x-www-form-urlencoded':
            form = await request.form()
            data = dict(form)
        else:
            logger.warning(f"Unsupported Media Type: {content_type}")
            result = SearchResponse.failure(
                error="Unsupported Media Type",
                code=415
            )
            response.status_code = result.code
            return result

        logger.info(f"Raw request data: {data}")
        logger.info(f"Parsed search_request: {search_request}")

        # Get client type
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')

        # Check for any valid authentication method
        auth_header = request.headers.get('Authorization')
        # If X-API-Key is provided but no Authorization header, construct the Authorization header
        if api_key and not auth_header:
            auth_header = f"APIKey {api_key}"

        if not any([
            auth_header and ('Bearer ' in auth_header or 'Session ' in auth_header or 'APIKey ' in auth_header),
            api_key,
            session_token
        ]):
            logger.warning("Authentication required. No valid auth header found.")
            result = SearchResponse.failure(
                error="Authentication required. Use Authorization header with Bearer/Session/APIKey, X-API-Key, or X-Session-Token",
                code=401
            )
            response.status_code = result.code
            return result
        # max_memories override with environment variable if present
        if env.get('SEARCH_MAX_MEMORIES'):
            max_memories = int(env.get('SEARCH_MAX_MEMORIES'))
        if env.get('SEARCH_MAX_NODES'):
            max_nodes = int(env.get('SEARCH_MAX_NODES'))

        # --- Optimized authentication using cached method ---
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')
        auth_start_time = time.time()
        try:
            async def _authenticate_with_client(httpx_client: httpx.AsyncClient):
                # Use optimized authentication that gets workspace_id, isQwenRoute, user_roles, user_workspace_ids, and resolves user in parallel
                # OPTIMIZATION: Skip schema fetching for search (only active patterns needed)
                if api_key and bearer_token:
                    # Developer provides API key + Bearer token for end user - use Bearer token for auth but developer's API key for Parse
                    return await get_user_from_token_optimized(
                        f"Bearer {bearer_token.credentials}",
                        client_type,
                        memory_graph,
                        api_key=api_key,
                        search_request=search_request,
                        httpx_client=httpx_client,
                        include_schemas=False,
                        url_enable_agentic_graph=enable_agentic_graph,
                    )
                if api_key and session_token:
                    return await get_user_from_token_optimized(
                        f"Session {session_token}",
                        client_type,
                        memory_graph,
                        api_key=api_key,
                        search_request=search_request,
                        httpx_client=httpx_client,
                        include_schemas=False,
                        url_enable_agentic_graph=enable_agentic_graph,
                    )
                if api_key:
                    return await get_user_from_token_optimized(
                        f"APIKey {api_key}",
                        client_type,
                        memory_graph,
                        search_request=search_request,
                        httpx_client=httpx_client,
                        include_schemas=False,
                        url_enable_agentic_graph=enable_agentic_graph,
                    )
                if bearer_token:
                    return await get_user_from_token_optimized(
                        f"Bearer {bearer_token.credentials}",
                        client_type,
                        memory_graph,
                        search_request=search_request,
                        httpx_client=httpx_client,
                        include_schemas=False,
                        url_enable_agentic_graph=enable_agentic_graph,
                    )
                if session_token:
                    return await get_user_from_token_optimized(
                        f"Session {session_token}",
                        client_type,
                        memory_graph,
                        search_request=search_request,
                        httpx_client=httpx_client,
                        include_schemas=False,
                        url_enable_agentic_graph=enable_agentic_graph,
                    )
                return await get_user_from_token_optimized(
                    auth_header,
                    client_type,
                    memory_graph,
                    search_request=search_request,
                    httpx_client=httpx_client,
                    include_schemas=False,
                    url_enable_agentic_graph=enable_agentic_graph,
                )

            httpx_client = getattr(request.app.state, "httpx_client", None)
            if httpx_client:
                auth_response = await _authenticate_with_client(httpx_client)
            else:
                async with httpx.AsyncClient() as httpx_client:
                    auth_response = await _authenticate_with_client(httpx_client)
        except ValueError as e:
            logger.error(f"Invalid authentication token: {e}")
            result = SearchResponse.failure(
                error="Invalid authentication token",
                code=401
            )
            response.status_code = result.code
            return result
        auth_end_time = time.time()
        logger.warning(f"Authentication timing: {(auth_end_time - auth_start_time) * 1000:.2f}ms")

        # Extract values from the optimized auth response
        user_id = auth_response.developer_id
        end_user_id = auth_response.end_user_id
        resolved_user_id = auth_response.end_user_id  # This is the resolved user ID
        session_token = auth_response.session_token
        user_info = auth_response.user_info
        api_key = auth_response.api_key
        workspace_id = auth_response.workspace_id
        is_qwen_route = auth_response.is_qwen_route
        user_roles = auth_response.user_roles
        user_workspace_ids = auth_response.user_workspace_ids
        updated_metadata = auth_response.updated_metadata

        # Augment user_workspace_ids with the authenticated workspace if not already present
        if workspace_id and (not user_workspace_ids or workspace_id not in user_workspace_ids):
            if not user_workspace_ids:
                user_workspace_ids = []
            user_workspace_ids.append(workspace_id)
            logger.info(f"Augmented user_workspace_ids with authenticated workspace: {workspace_id}")

        logger.info(
            "Auth response multi-tenant context: org_id=%s, namespace_id=%s, workspace_id=%s",
            getattr(auth_response, "organization_id", None),
            getattr(auth_response, "namespace_id", None),
            workspace_id,
        )

        # Extract multi-tenant context using utility function
        from services.multi_tenant_utils import extract_multi_tenant_context, apply_multi_tenant_scoping_to_search_request
        auth_context = extract_multi_tenant_context(auth_response)
        cached_schema = auth_response.cached_schema  # Pre-fetched cached schema patterns for agentic search optimization
        # Note: No longer fetching UserGraphSchema for search - only ActiveNodeRel patterns matter for agentic search

        # Set developer_id from user_info
        developer_id = user_info.get("developer_id") if user_info and "developer_id" in user_info else user_id

        # Use search_request model instead of parsing raw request
        query = search_request.query
        rank_results = search_request.rank_results
        logger.info(f"search_request.query: {query}")
        logger.info(f"search_request.rank_results: {rank_results}")
        logger.info(
            "Incoming SearchRequest org_id=%s, namespace_id=%s",
            getattr(search_request, "organization_id", None),
            getattr(search_request, "namespace_id", None),
        )
        
        # Setup reranking configuration
        # Use reranking_config from SearchRequest if provided, otherwise fall back to rank_results boolean
        reranking_config = getattr(search_request, 'reranking_config', None)
        if not reranking_config and rank_results:
            reranking_config = RerankingConfig(
                reranking_enabled=True,
                reranking_provider=RerankingProvider.OPENAI,  # Default to OpenAI for backward compatibility
                reranking_model=env.get("LLM_MODEL", "gpt-5-nano")
            )
        # Early validation of query
        if not query.strip():
            logger.warning("Query is empty or whitespace. Returning 400.")
            result = SearchResponse.failure(
                error="Invalid query",
                code=400
            )
            response.status_code = result.code
            return result
        
        project_id = None
        context = ''
        relation_type = ''
        
        # Use resolved values from optimized authentication
        if updated_metadata:
            metadata = updated_metadata
            # Preserve developer-provided customMetadata filters by merging them
            try:
                if getattr(search_request, "metadata", None) and getattr(search_request.metadata, "customMetadata", None):
                    incoming_custom = search_request.metadata.customMetadata or {}
                    # Ensure target has a dict to merge into
                    if getattr(metadata, "customMetadata", None) is None:
                        metadata.customMetadata = {}
                    # Merge without dropping any developer-provided keys
                    for key, value in incoming_custom.items():
                        metadata.customMetadata[key] = value
            except Exception:
                # Best-effort merge; if anything goes wrong, proceed without blocking search
                pass
        else:
            # Fallback: build metadata from search request if not resolved in auth
            metadata = search_request.metadata or MemoryMetadata()
            if getattr(search_request, "user_id", None) is not None:
                metadata.user_id = search_request.user_id
            if getattr(search_request, "external_user_id", None) is not None:
                metadata.external_user_id = search_request.external_user_id
            if getattr(metadata, "user_id", None) is None and getattr(metadata, "external_user_id", None) is None:
                metadata.user_id = user_id
            # Remove any None fields from metadata
            metadata_dict = metadata.model_dump(exclude_none=True)
            metadata = MemoryMetadata(**metadata_dict)

        # Apply multi-tenant scoping using utility function
        metadata = apply_multi_tenant_scoping_to_search_request(search_request, auth_context)

        # Ensure SearchRequest carries scoped org/namespace IDs for downstream ACL filters
        if getattr(search_request, "organization_id", None) is None and auth_context.get("organization_id"):
            setattr(search_request, "organization_id", auth_context["organization_id"])
            logger.info(f"Propagated organization_id to SearchRequest: {search_request.organization_id}")
        if getattr(search_request, "namespace_id", None) is None and auth_context.get("namespace_id"):
            setattr(search_request, "namespace_id", auth_context["namespace_id"])
            logger.info(f"Propagated namespace_id to SearchRequest: {search_request.namespace_id}")
        logger.info(
            "Post-scoping SearchRequest org_id=%s, namespace_id=%s",
            getattr(search_request, "organization_id", None),
            getattr(search_request, "namespace_id", None),
        )
        logger.info(
            "Metadata after scoping: org_id=%s, namespace_id=%s, custom_keys=%s",
            getattr(metadata, "organization_id", None),
            getattr(metadata, "namespace_id", None),
            list((getattr(metadata, "customMetadata", None) or {}).keys()),
        )

        logger.info(f"Resolved end_user_id: {end_user_id}, developer_id: {developer_id}")
        logger.info(f"is_qwen_route: {is_qwen_route}")
        logger.info(f"workspace_id: {workspace_id}")
        
        # Check rate limits (CLOUD ONLY - OSS has no limits)
        from config import get_features
        features = get_features()
        
        rate_limit_start_time = time.time()
        user_instance = User(user_id)
        
        # Start rate limit check as a background task to run in parallel with retrieval
        rate_limit_task = None
        rate_limit_completion_time = None
        
        # Only check rate limits in cloud edition (OSS is self-hosted, no limits needed)
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
            
            # Create a wrapper to track when the rate limit task actually completes
            # Extract features from search_request for variable cost calculation
            from models.operation_types import MemoryOperationType
            # enable_agentic_graph already extracted earlier from search_request
            rank_results = getattr(search_request, 'rank_results', False)
            
            async def rate_limit_with_timing():
                nonlocal rate_limit_completion_time
                result = await user_instance.check_interaction_limits_fast(
                    interaction_type='mini',
                    memory_graph=memory_graph,
                    operation=MemoryOperationType.SEARCH_MEMORY,
                    enable_agentic_graph=enable_agentic_graph,
                    enable_rank_results=rank_results,
                    api_key_id=api_key_id,
                    organization_id=organization_id,
                    namespace_id=namespace_id,
                    defer_usage_tracking=True
                )
                rate_limit_completion_time = time.time()
                return result
            rate_limit_task = asyncio.create_task(rate_limit_with_timing())
        else:
            if not features.is_cloud:
                logger.info("OSS edition - bypassing rate limits (self-hosted)")
            else:
                logger.info("EVALmetrics enabled - bypassing rate limits")
        
        external_user_id = search_request.user_id if search_request.user_id else user_id
        logger.info(f"external_user_id: {external_user_id}")
        
        # Use workspace_id from enhanced authentication
        logger.info(f"Using workspace_id from enhanced auth: {workspace_id}")
        logger.info(f"Using user_roles from enhanced auth: {len(user_roles) if user_roles else 0} roles")
        logger.info(f"Using user_workspace_ids from enhanced auth: {len(user_workspace_ids) if user_workspace_ids else 0} workspaces")
        
        # Start retrieval timing
        retrieval_start_time = time.time()
        
        # --- Optimized execution: Neo4j session created only when needed ---
        # Run rate limit check and memory search in parallel
        memory_search_start_time = time.time()
        pre_search_time_ms = (memory_search_start_time - search_start_time) * 1000
        logger.warning(f"Pre-search overhead (auth/parse/etc): {pre_search_time_ms:.2f}ms")
        
        # Create tasks for parallel execution
        tasks = []
        
        # Add memory search task (Neo4j session will be created internally if needed)
        # Normalize is_qwen_route: if None or False, default to False
        if is_qwen_route is None:
            is_qwen_route = False
            logger.info(f"is_qwen_route was None, defaulting to False")
        
        # legacy_route is the opposite of is_qwen_route
        legacy_route = not is_qwen_route
        logger.info(f"Setting legacy_route for find_related_memory_items_async: {legacy_route} (is_qwen_route: {is_qwen_route})")
        
        logger.warning(f"🚀 CALLING find_related_memory_items_async with skip_neo={skip_neo}")
        print(f"🚀 CALLING find_related_memory_items_async with skip_neo={skip_neo}")  # Also print to stdout
        
        memory_task = memory_graph.find_related_memory_items_async(
            session_token, 
            query, 
            resolved_user_id,
            chat_gpt, 
            neo_session=None,  # Let the method create its own session when needed
            metadata=metadata, 
            relation_type=relation_type, 
            project_id=project_id, 
            skip_neo=skip_neo,
            user_workspace_ids=user_workspace_ids,
            user_roles=user_roles,
            reranking_config=reranking_config,
            api_key=api_key,
            context=context,
            top_k=max_memories,
            top_k_neo=max_nodes,
            legacy_route=legacy_route,
            cached_schema=cached_schema,  # Pass pre-fetched cached schema patterns for agentic search optimization
            search_request=search_request
            # Note: No longer passing UserGraphSchema - only ActiveNodeRel patterns needed for search
        )
        tasks.append(memory_task)
        
        # Add rate limit task if needed
        if rate_limit_task:
            tasks.append(rate_limit_task)
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Results: {results}")
        
        # Extract results and handle exceptions
        if rate_limit_task:
            # First result is memory search, second is rate limit
            relevant_items = results[0]
            limit_check = results[1]
        else:
            # Only memory search result
            relevant_items = results[0]
            limit_check = None
        
        # Check if any result is an exception and return failure response
        if isinstance(relevant_items, Exception):
            logger.error(f"Memory search failed: {relevant_items}")
            result = SearchResponse.failure(
                error=f"Memory search failed: {str(relevant_items)}",
                code=500
            )
            response.status_code = result.code
            return result
        if limit_check and isinstance(limit_check, Exception):
            logger.error(f"Rate limit check failed: {limit_check}")
            result = SearchResponse.failure(
                error=f"Rate limit check failed: {str(limit_check)}",
                code=500
            )
            response.status_code = result.code
            return result
        
        memory_search_end_time = time.time()
        logger.warning(f"Memory search timing: {(memory_search_end_time - memory_search_start_time) * 1000:.2f}ms")
        post_search_time_ms = (time.time() - memory_search_end_time) * 1000
        logger.warning(f"Post-search overhead (response/build/logging): {post_search_time_ms:.2f}ms")
        if isinstance(relevant_items, Exception):
            logger.error(f"Error in memory search: {relevant_items}", exc_info=True)
            result = SearchResponse.failure(
                error=f"Error in memory search",
                code=500
            )
            response.status_code = result.code
            return result
        # Optionally, return an error response here
        else:
            if relevant_items is not None and relevant_items.memory_items is not None:
                logger.info(f"Relevant items length: {len(relevant_items.memory_items)}")

        # End retrieval timing
        retrieval_end_time = time.time()
        retrieval_latency_ms = (retrieval_end_time - retrieval_start_time) * 1000
        logger.warning(f"Total retrieval timing: {retrieval_latency_ms:.2f}ms")

        # Check rate limit result (it ran in parallel with retrieval)
        if limit_check and not isinstance(limit_check, Exception):
            limit_response, status_code, is_error = limit_check
            if is_error:
                logger.warning(f"Rate limit error: {limit_response}")
                # Extract error message from the response dictionary
                error_message = limit_response.get('error', 'Subscription required') if isinstance(limit_response, dict) else str(limit_response)
                
                # Check if this is specifically an interaction tracking error
                if "Unable to update usage record" in error_message:
                    logger.warning(f"Interaction tracking failed but continuing with search: {error_message}")
                    # Log the issue but don't fail the search - this is likely a data consistency issue
                    # The search results are valid, we just couldn't track usage
                else:
                    # For other rate limit errors (subscription issues, etc.), still fail
                    result = SearchResponse.failure(
                        error=error_message,
                        code=status_code,
                        details=limit_response  # Store full response in details
                    )
                    response.status_code = result.code
                    return result
            else:
                logger.info(f"Rate limit check success with message: {limit_response}")
        elif isinstance(limit_check, Exception):
            logger.error(f"Error in rate limit check: {limit_check}")
            # Continue with the request even if rate limit check fails
        
        # Rate limit timing should now be the same as total retrieval timing since they ran in parallel
        rate_limit_end_time = time.time()
        rate_limit_actual_time = (rate_limit_end_time - rate_limit_start_time) * 1000
        logger.warning(f"Rate limit check timing: {rate_limit_actual_time:.2f}ms (parallel with retrieval)")
        
        # Show actual completion time if available
        if rate_limit_completion_time:
            rate_limit_completion_ms = (rate_limit_completion_time - rate_limit_start_time) * 1000
            logger.warning(f"Rate limit actual completion: {rate_limit_completion_ms:.2f}ms")
            logger.warning(f"Parallel execution benefit: {max(0, rate_limit_completion_ms - retrieval_latency_ms):.2f}ms saved")
        else:
            logger.warning(f"Parallel execution benefit: {max(0, rate_limit_actual_time - retrieval_latency_ms):.2f}ms saved")

        # Check if we have any actual results
        if not relevant_items.memory_items and not relevant_items.neo_nodes:
            logger.info("No relevant items found")
            result = SearchResponse.failure(
                error="No relevant items found",
                code=404
            )
            response.status_code = result.code
            return result

        # Normalize relevance scores and sorting for memory items
        rank_results = getattr(search_request, 'rank_results', False)
        memory_items_full = list(relevant_items.memory_items or [])
        similarity_scores_by_id = relevant_items.similarity_scores_by_id or {}
        confidence_scores = relevant_items.confidence_scores or []
        has_rerank_scores = bool(confidence_scores) and len(confidence_scores) == len(memory_items_full)

        # Debug logging for similarity score matching
        logger.info(f"DEBUG: similarity_scores_by_id has {len(similarity_scores_by_id)} entries")
        logger.info(f"DEBUG: similarity_scores_by_id keys (first 10): {list(similarity_scores_by_id.keys())[:10]}")
        logger.info(f"DEBUG: similarity_scores_by_id values (first 10): {list(similarity_scores_by_id.values())[:10]}")
        logger.info(f"DEBUG: memory_items_full has {len(memory_items_full)} items")
        for i, mem in enumerate(memory_items_full[:5]):
            mem_id = getattr(mem, "memoryId", None) or getattr(mem, "id", None) or getattr(mem, "objectId", None)
            chunk_ids = getattr(mem, "memoryChunkIds", None) or []
            logger.info(f"DEBUG: memory[{i}] memoryId={mem_id}, memoryChunkIds={chunk_ids[:3] if chunk_ids else []}")

        def _score_from_similarity(mem):
            mem_id = getattr(mem, "memoryId", None) or getattr(mem, "id", None) or getattr(mem, "objectId", None)
            # Direct match on memory ID
            if mem_id and mem_id in similarity_scores_by_id:
                return similarity_scores_by_id.get(mem_id)
            # Check chunk IDs from the memory
            chunk_ids = getattr(mem, "memoryChunkIds", None) or []
            for chunk_id in chunk_ids:
                if chunk_id in similarity_scores_by_id:
                    return similarity_scores_by_id.get(chunk_id)
            # Check if any similarity score key starts with this memory ID (e.g., memoryId_0, memoryId_grouped_0)
            if mem_id:
                for score_key, score_val in similarity_scores_by_id.items():
                    # Match pattern: memoryId_<number> or memoryId_grouped_<number>
                    if score_key.startswith(mem_id + '_') or score_key.startswith(mem_id + '_grouped'):
                        return score_val
                # Also check if the score key (with suffixes stripped) matches the memory ID
                for score_key, score_val in similarity_scores_by_id.items():
                    # Strip _grouped and _<number> suffixes from score_key for comparison
                    base_score_key = re.sub(r'(_grouped)?(_\d+)?$', '', score_key)
                    if base_score_key == mem_id:
                        return score_val
            return None

        # =============================================================================
        # RELEVANCE SCORING - Research-backed multi-signal fusion
        # =============================================================================
        # References:
        #   - BM25/IR: log1p() normalization for frequency (prevents popularity bias)
        #   - RRF (Cormack et al. 2009): score = Σ 1/(k + rank_i) for robust rank fusion
        #   - Time decay: exp(-λ * age) with configurable half-life
        #   - Multi-signal fusion outperforms single signals (LTR research)
        #
        # SCORES COMPUTED AT SEARCH TIME:
        #   1. similarity_score: Cosine similarity from vector search (0-1)
        #   2. popularity_score: 0.5*cacheConf + 0.5*citationConf (0-1)
        #   3. recency_score: exp(-0.05 * days_since_access) (0-1, half-life ~14 days)
        #   4. reranker_score: Cross-encoder or LLM relevance (0-1, only if rank_results=True)
        #   5. relevance_score: Final combined score for ranking (0-1)
        # =============================================================================
        import math
        from datetime import datetime, timezone

        # Determine reranker type from config
        reranking_config = getattr(search_request, 'reranking_config', None)
        reranking_provider = None
        if reranking_config and reranking_config.reranking_enabled:
            reranking_provider = reranking_config.reranking_provider

        # Helper: Determine if provider is LLM-based or cross-encoder
        def _is_llm_reranker(provider) -> bool:
            """OpenAI models are LLM-based; Cohere uses cross-encoder"""
            if provider is None:
                return False
            from models.memory_models import RerankingProvider
            return provider == RerankingProvider.OPENAI

        # RRF constant (Cormack et al. 2009)
        RRF_K = 60

        # ---------------------------------------------------------------------
        # Step 1: similarity_score - Cosine similarity from vector search
        # ---------------------------------------------------------------------
        sim_count = 0
        for mem in memory_items_full:
            sim_score = _score_from_similarity(mem)
            if sim_score is not None:
                mem.similarity_score = float(sim_score)
                sim_count += 1
        logger.info(f"SCORES: similarity_score assigned to {sim_count}/{len(memory_items_full)} memories")

        # ---------------------------------------------------------------------
        # Step 2: popularity_score - From stored cache/citation EMA fields
        # Formula: 0.5 * cacheConfidenceWeighted30d + 0.5 * citationConfidenceWeighted30d
        # Uses log1p normalization if raw counts, but confidence fields are already 0-1
        # ---------------------------------------------------------------------
        pop_count = 0
        for mem in memory_items_full:
            cache_conf = getattr(mem, 'cacheConfidenceWeighted30d', None) or 0.0
            cite_conf = getattr(mem, 'citationConfidenceWeighted30d', None) or 0.0
            # Both confidence fields are already in reasonable 0-1ish range
            # Normalize to ensure 0-1 output (cap at 1.0)
            popularity = 0.5 * min(float(cache_conf), 1.0) + 0.5 * min(float(cite_conf), 1.0)
            mem.popularity_score = float(min(popularity, 1.0))
            if cache_conf > 0 or cite_conf > 0:
                pop_count += 1
        logger.info(f"SCORES: popularity_score computed for {pop_count}/{len(memory_items_full)} memories with data")

        # ---------------------------------------------------------------------
        # Step 3: recency_score - Exponential time decay
        # Formula: exp(-λ * days_since_last_access) where λ = 0.05 (half-life ~14 days)
        # Half-life calculation: 0.5 = exp(-λ * t) => t = ln(2)/λ ≈ 13.9 days
        # ---------------------------------------------------------------------
        RECENCY_LAMBDA = 0.05  # Decay rate (half-life ~14 days)
        now = datetime.now(timezone.utc)
        rec_count = 0
        for mem in memory_items_full:
            last_accessed = getattr(mem, 'lastAccessedAt', None)
            if last_accessed:
                try:
                    if isinstance(last_accessed, str):
                        last_accessed = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
                    elif isinstance(last_accessed, dict) and 'iso' in last_accessed:
                        last_accessed = datetime.fromisoformat(last_accessed['iso'].replace('Z', '+00:00'))

                    if last_accessed.tzinfo is None:
                        last_accessed = last_accessed.replace(tzinfo=timezone.utc)

                    days_since = (now - last_accessed).total_seconds() / 86400.0
                    recency = math.exp(-RECENCY_LAMBDA * max(days_since, 0))
                    mem.recency_score = float(recency)
                    rec_count += 1
                except Exception as e:
                    logger.debug(f"Failed to compute recency for memory: {e}")
                    mem.recency_score = 0.5  # Default to mid-range
            else:
                mem.recency_score = 0.5  # Default for memories without lastAccessedAt
        logger.info(f"SCORES: recency_score computed for {rec_count}/{len(memory_items_full)} memories with lastAccessedAt")

        # ---------------------------------------------------------------------
        # Step 4: reranker_score - Only when rank_results=True
        # Cross-encoder (Cohere): relevance_score directly
        # LLM (GPT-5-nano): score normalized from 1-10 to 0-1, plus confidence
        # ---------------------------------------------------------------------
        if rank_results and has_rerank_scores:
            is_llm = _is_llm_reranker(reranking_provider)
            reranker_type_str = "llm" if is_llm else "cross_encoder"

            for i, mem in enumerate(memory_items_full):
                if i < len(confidence_scores):
                    raw_score = confidence_scores[i] or 0.0

                    if is_llm:
                        # LLM returns score 1-10, normalize to 0-1
                        # confidence_scores contains the normalized confidence (0-1)
                        mem.reranker_score = float(min(raw_score, 1.0))
                        mem.reranker_confidence = float(min(raw_score, 1.0))
                    else:
                        # Cross-encoder returns 0-1 directly
                        mem.reranker_score = float(min(raw_score, 1.0))
                        mem.reranker_confidence = float(min(raw_score, 1.0))

                    mem.reranker_type = reranker_type_str

            logger.info(f"SCORES: reranker_score ({reranker_type_str}) assigned to {len(memory_items_full)} memories")

        # ---------------------------------------------------------------------
        # Step 5: relevance_score - Final combined score for ranking
        # ---------------------------------------------------------------------
        # rank_results=False:
        #   relevance_score = 0.60 * similarity + 0.25 * popularity + 0.15 * recency
        #
        # rank_results=True:
        #   Use RRF to combine similarity and reranker rankings, then boost
        #   rrf_score = 1/(k + sim_rank) + 1/(k + reranker_rank)
        #   relevance_score = normalize(rrf) * (1 + 0.1*popularity + 0.05*recency)
        # ---------------------------------------------------------------------

        if rank_results and has_rerank_scores:
            # Build ranking lists for RRF
            # Rank 1: Semantic similarity ranking
            semantic_ranks = {}
            sorted_by_sim = sorted(
                [(i, getattr(mem, 'similarity_score', None) or 0.0) for i, mem in enumerate(memory_items_full)],
                key=lambda x: x[1], reverse=True
            )
            for rank, (idx, _) in enumerate(sorted_by_sim, start=1):
                semantic_ranks[idx] = rank

            # Rank 2: Reranker score ranking
            reranker_ranks = {}
            sorted_by_rerank = sorted(
                [(i, getattr(mem, 'reranker_score', None) or 0.0) for i, mem in enumerate(memory_items_full)],
                key=lambda x: x[1], reverse=True
            )
            for rank, (idx, _) in enumerate(sorted_by_rerank, start=1):
                reranker_ranks[idx] = rank

            # Compute RRF-based relevance_score with behavioral boost
            max_rrf = 2.0 / (RRF_K + 1)  # Max possible RRF with 2 lists

            for i, mem in enumerate(memory_items_full):
                sem_rank = semantic_ranks.get(i, len(memory_items_full))
                rerank_rank = reranker_ranks.get(i, len(memory_items_full))

                # RRF formula: sum of reciprocal ranks
                rrf_score = (1.0 / (RRF_K + sem_rank)) + (1.0 / (RRF_K + rerank_rank))
                normalized_rrf = rrf_score / max_rrf  # Normalize to 0-1

                # Boost with popularity and recency signals
                pop = getattr(mem, 'popularity_score', 0.0) or 0.0
                rec = getattr(mem, 'recency_score', 0.5) or 0.5
                boost = 1.0 + 0.1 * pop + 0.05 * rec

                mem.relevance_score = float(min(normalized_rrf * boost, 1.0))

            logger.info(f"SCORES: relevance_score (RRF + boost) computed for {len(memory_items_full)} memories")
        else:
            # rank_results=False: Simple weighted combination
            # relevance_score = 0.60 * similarity + 0.25 * popularity + 0.15 * recency
            for mem in memory_items_full:
                sim = getattr(mem, 'similarity_score', None) or 0.0
                pop = getattr(mem, 'popularity_score', None) or 0.0
                rec = getattr(mem, 'recency_score', None) or 0.5

                relevance = 0.60 * sim + 0.25 * pop + 0.15 * rec
                mem.relevance_score = float(min(relevance, 1.0))

            logger.info(f"SCORES: relevance_score (weighted) computed for {len(memory_items_full)} memories")

        # ---------------------------------------------------------------------
        # Sort memories by relevance_score (final ranking)
        # ---------------------------------------------------------------------
        if memory_items_full:
            def _sort_key(mem):
                score = getattr(mem, "relevance_score", None)
                return score if score is not None else -1.0
            memory_items_full = sorted(memory_items_full, key=_sort_key, reverse=True)
            relevant_items.memory_items = memory_items_full
        
        # Use max_memories from request body if present, otherwise use query param
        
        max_memories = getattr(search_request, 'max_memories', None) or max_memories
        max_nodes = getattr(search_request, 'max_nodes', None) or max_nodes

        # Log before truncation
        logger.info(f"Number of memory_items before truncation: {len(relevant_items.memory_items)}")
        logger.info(f"Number of neo_nodes before truncation: {len(relevant_items.neo_nodes)}")
        logger.info(f"Truncating to max_memories={max_memories}, max_nodes={max_nodes}")

        memory_items = relevant_items.memory_items[:max_memories]
        neo_nodes = relevant_items.neo_nodes[:max_nodes]

        # Log after truncation
        logger.info(f"Number of memory_items after truncation: {len(memory_items)}")
        logger.info(f"Number of neo_nodes after truncation: {len(neo_nodes)}")
        
        # Count legacy vs new memories (legacy = no organization_id or namespace_id)
        legacy_count = 0
        new_count = 0
        for mem in memory_items:
            has_org = hasattr(mem, 'organization_id') and mem.organization_id is not None
            has_ns = hasattr(mem, 'namespace_id') and mem.namespace_id is not None
            if has_org or has_ns:
                new_count += 1
            else:
                legacy_count += 1
        
        logger.warning(f"📊 SEARCH RESULTS SUMMARY: Total memories returned: {len(memory_items)} (Legacy: {legacy_count}, New: {new_count}), Nodes: {len(neo_nodes)}")

        # Keep memory_source_info intact for background logging (predicted grouping detection)
        # Only strip heavy context from response
        relevant_items.neo_context = None
        # Update items after truncation
        relevant_items.memory_items = memory_items
        relevant_items.neo_nodes = neo_nodes

        logger.info(f"Before calling SearchResult.from_internal")

        # Create schema mapping for custom nodes using cached auth data (zero additional calls!)
        schema_mapping = {}
        user_schemas = getattr(auth_response, 'user_schemas', None) or []
        
        if user_schemas:
            # Build mapping from node label to schema ID using already-fetched schemas
            for schema in user_schemas:
                # Handle both UserGraphSchema objects and dict representations
                if hasattr(schema, 'node_types') and hasattr(schema, 'id'):
                    # UserGraphSchema object
                    for node_type in schema.node_types.values():
                        schema_mapping[node_type.name] = schema.id
                elif isinstance(schema, dict):
                    # Dict representation
                    schema_id = schema.get('id')
                    node_types = schema.get('node_types', {})
                    if schema_id and node_types:
                        for node_type in node_types.values():
                            if isinstance(node_type, dict):
                                schema_mapping[node_type.get('name')] = schema_id
                            else:
                                schema_mapping[node_type.name] = schema_id
                        
            logger.info(f"🔗 SCHEMA MAPPING (CACHED): Built mapping for {len(schema_mapping)} custom node types from {len(user_schemas)} cached schemas")
            logger.info(f"🔗 SCHEMA MAPPING (CACHED): {schema_mapping}")
        else:
            logger.info(f"🔗 SCHEMA MAPPING (CACHED): No user schemas available in auth response")

        # Convert RelatedMemoryResult to SearchResult using from_internal
        search_result_conversion_start_time = time.time()
        logger.info(f"🔍 BEFORE CONVERSION: relevant_items.neo_nodes count: {len(relevant_items.neo_nodes)}")
        logger.info(f"🔍 BEFORE CONVERSION: schema_mapping: {schema_mapping}")
        search_result = SearchResult.from_internal(relevant_items, schema_mapping=schema_mapping)
        search_result_conversion_end_time = time.time()
        logger.info(f"SearchResult conversion timing: {(search_result_conversion_end_time - search_result_conversion_start_time) * 1000:.2f}ms")
        logger.info(f"🔍 AFTER CONVERSION: search_result.nodes count: {len(search_result.nodes)}")
        logger.info(f"🔍 AFTER CONVERSION: search_result.memories count: {len(search_result.memories)}")

        # Success response
        response_creation_start_time = time.time()
        result = SearchResponse.success(
            data=search_result,
            code=200,
            search_id=search_id
        )
        
        # TEMPORARY DEBUG: Add debug info to response
        if hasattr(result.data, '__dict__'):
            result.data.__dict__['debug_agentic_params'] = {
                'enable_agentic_graph': enable_agentic_graph,
                'skip_neo': skip_neo
            }
        response.status_code = result.code
        response_creation_end_time = time.time()
        logger.warning(f"Response creation timing: {(response_creation_end_time - response_creation_start_time) * 1000:.2f}ms")
        
        # Convert to TOON format if requested (30-60% token reduction for LLMs)
        if response_format == ResponseFormat.TOON:
            try:
                # Convert SearchResponse to dict, then encode to TOON format
                result_dict = result.model_dump(mode='json', exclude_none=True)
                
                # Strip empty values (empty strings, empty lists, empty dicts) to further reduce token usage
                result_dict_cleaned = strip_empty_values(result_dict)
                
                # Calculate savings from cleaning
                original_size = len(json.dumps(result_dict))
                cleaned_size = len(json.dumps(result_dict_cleaned))
                cleaning_reduction = 100 * (1 - cleaned_size / original_size) if original_size > 0 else 0
                logger.info(f"Empty value cleaning: {original_size} chars → {cleaned_size} chars ({cleaning_reduction:.1f}% reduction)")
                
                toon_string = toon_encode(result_dict_cleaned, {
                    "indent": 2,
                    "delimiter": ",",
                    "lengthMarker": "#"  # Add # prefix for validation
                })
                
                total_reduction = 100 * (1 - len(toon_string) / original_size) if original_size > 0 else 0
                logger.info(f"Converted response to TOON format. JSON size: {original_size} chars, TOON size: {len(toon_string)} chars ({total_reduction:.1f}% total reduction)")
                
                # Return TOON as plain text with appropriate content-type
                return PlainTextResponse(
                    content=toon_string,
                    status_code=200,
                    media_type="text/plain; charset=utf-8",
                    headers={"X-Content-Format": "toon"}
                )
            except Exception as e:
                logger.error(f"Failed to encode response as TOON: {e}")
                # Fall back to JSON if TOON encoding fails
                logger.warning("Falling back to JSON response due to TOON encoding error")
        
        # Fire-and-forget logging tasks to avoid blocking response timing (tests + clients)
        background_task_start_time = time.time()
        disable_bg_tasks = os.getenv("PERF_TEST_DISABLE_BG_TASKS", "").lower() in {"1", "true", "yes"}

        if disable_bg_tasks:
            logger.info("PERF_TEST_DISABLE_BG_TASKS enabled - skipping background tasks for search_v1")
        else:
            if workspace_id:  # Only log if we have a workspace
                try:
                    asyncio.create_task(
                        query_log_service.create_query_and_retrieval_logs_background(
                query=query,
                search_request=search_request,
                metadata=metadata,
                resolved_user_id=resolved_user_id,
                workspace_id=workspace_id,
                relevant_items=relevant_items,
                retrieval_latency_ms=retrieval_latency_ms,
                search_start_time=search_start_time,
                session_token=session_token,
                api_key=api_key,
                client_type=client_type,
                chat_gpt=chat_gpt,
                search_id=search_id  # Pass the generated search_id
            )
                    )
                except Exception as e:
                    logger.error(f"Error scheduling query log background task: {e}")

            # Add telemetry logging as fire-and-forget task (edition-aware)
        # Uses TelemetryService which handles OSS (PostHog) vs Cloud (Amplitude) automatically
        async def log_search_telemetry():
            """Background task for telemetry"""
            try:
                from core.services.telemetry import get_telemetry
                telemetry = get_telemetry()
                await telemetry.track(
                    "search",
                    {
                        "client_type": client_type,
                        "has_results": len(memory_items) > 0 if memory_items else False,
                        "result_count": len(memory_items) if memory_items else 0,
                        "neo_node_count": len(neo_nodes) if neo_nodes else 0,
                        "retrieval_latency_ms": retrieval_latency_ms,
                        "enable_agentic_graph": search_request.enable_agentic_graph,
                        "api_key": api_key,  # Track which API key is used (anonymized in OSS)
                    },
                    user_id=resolved_user_id,  # End user
                    developer_id=developer_id  # API key owner
                )
            except Exception as e:
                logger.error(f"Error tracking search: {e}")
        
            try:
                asyncio.create_task(log_search_telemetry())
            except Exception as e:
                logger.error(f"Error scheduling telemetry background task: {e}")
        
        background_task_end_time = time.time()
        logger.warning(f"Background task setup timing: {(background_task_end_time - background_task_start_time) * 1000:.2f}ms")

        # Log total processing time
        total_processing_time_ms = (time.time() - search_start_time) * 1000
        logger.warning(f"Total search processing time: {total_processing_time_ms:.2f}ms")

        return result

    except ValueError as ve:
        logger.error(f"ValueError in search_v1: {ve}")
        logger.error(f"Request data that caused error: {data}")
        result = SearchResponse.failure(
            error=str(ve),
            code=400
        )
        response.status_code = result.code
        return result
    except Exception as e:
        logger.error(f"Error in get_memory: {e}", exc_info=True)
        result = SearchResponse.failure(
            error=str(e),
            code=500
        )
        response.status_code = result.code
        return result


async def _log_amplitude_event_background(
    event_type: str,
    user_info: Dict[str, Any],
    client_type: str,
    amplitude_client,
    logger,
    api_key: Optional[str],
    user_id: str,
    end_user_id: str,
    extra_properties: Optional[Dict[str, Any]] = None
) -> None:
    """
    Background function to log Amplitude events without blocking the response
    """
    try:
        if extra_properties is None:
            extra_properties = {}
            
        if api_key:
            extra_properties['is_developer'] = True
            extra_properties['developer_id'] = user_id
        else:
            extra_properties['is_developer'] = False
        
        success = await log_amplitude_event(
            event_type=event_type,
            user_info=user_info,
            client_type=client_type,
            amplitude_client=amplitude_client,
            logger=logger,
            extra_properties=extra_properties,
            end_user_id=end_user_id
        )
        
        if success:
            logger.info(f"Amplitude event logged successfully for {event_type}")
        else:
            logger.warning(f"Failed to log Amplitude event for {event_type}")

    except Exception as e:
        logger.error(f"Error logging event to Amplitude: {e}")

    