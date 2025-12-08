from fastapi import APIRouter, HTTPException, Request, Depends, Response, BackgroundTasks, Header, Query, Body, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
from typing import Optional, Dict, Any, List
import json
import uuid
import os
from datetime import datetime
from os import environ as env
from dotenv import find_dotenv, load_dotenv
import time
import httpx

from memory.memory_graph import MemoryGraph
from models.user_schemas import (
    UserGraphSchema, SchemaResponse, SchemaListResponse, SchemaStatus
)
from services.schema_service import SchemaService, get_schema_service
from services.auth_utils import get_user_from_token_optimized
from services.user_utils import User
from services.logger_singleton import LoggerSingleton
from services.utils import log_amplitude_event, get_memory_graph
from amplitude import Amplitude

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

router = APIRouter(prefix="/schemas", tags=["Schema Management"])

@router.post("",
    response_model=SchemaResponse,
    responses={
        201: {
            "model": SchemaResponse,
            "description": "Schema created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": {
                            "id": "schema_123456789",
                            "name": "E-commerce Schema",
                            "description": "Schema for e-commerce operations",
                            "version": "1.0.0",
                            "status": "active",
                            "node_types": {
                                "Product": {
                                    "name": "Product",
                                    "label": "Product",
                                    "description": "E-commerce product",
                                    "properties": {
                                        "name": {
                                            "type": "string", 
                                            "required": True,
                                            "description": "Product name, typically 2-4 words describing the item (e.g., 'iPhone 15 Pro', 'Nike Running Shoes')"
                                        },
                                        "price": {
                                            "type": "float", 
                                            "required": True,
                                            "description": "Product price in USD, as a decimal number (e.g., 999.99, 29.95)"
                                        },
                                        "category": {
                                            "type": "string",
                                            "required": True,
                                            "description": "Main product category - choose the most appropriate category for this item",
                                            "enum_values": ["electronics", "clothing", "books", "home", "sports"]
                                        },
                                        "condition": {
                                            "type": "string",
                                            "required": False,
                                            "description": "Physical condition of the product - use 'new' for brand new items, 'like_new' for barely used",
                                            "enum_values": ["new", "like_new", "good", "fair", "poor"],
                                            "default": "new"
                                        },
                                        "in_stock": {
                                            "type": "boolean",
                                            "required": True,
                                            "description": "Availability status - true if currently available for purchase, false if out of stock"
                                        },
                                        "sku": {
                                            "type": "string",
                                            "required": True,
                                            "description": "Stock keeping unit - exact alphanumeric code for inventory tracking",
                                            "enum_values": ["SKU-001", "SKU-002", "SKU-003", "SKU-004", "SKU-005"]
                                        }
                                    },
                                    "required_properties": ["name", "price", "category", "in_stock", "sku"],
                                    "unique_identifiers": ["name", "sku"],
                                    "color": "#e74c3c"
                                }
                            },
                            "relationship_types": {
                                "PURCHASED": {
                                    "name": "PURCHASED",
                                    "allowed_source_types": ["Customer"],
                                    "allowed_target_types": ["Product"]
                                }
                            }
                        },
                        "error": None,
                        "code": 201
                    }
                }
            }
        },
        400: {
            "model": SchemaResponse,
            "description": "Invalid schema definition",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "data": None,
                        "error": "Node type 'Memory' conflicts with system schema",
                        "code": 400
                    }
                }
            }
        },
        401: {
            "model": SchemaResponse,
            "description": "Unauthorized",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "data": None,
                        "error": "Missing or invalid authentication",
                        "code": 401
                    }
                }
            }
        },
        500: {
            "model": SchemaResponse,
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "data": None,
                        "error": "Internal server error",
                        "code": 500
                    }
                }
            }
        }
    },
    description="""Create a new user-defined graph schema.
    
    This endpoint allows users to define custom node types and relationships for their knowledge graph.
    The schema will be validated and stored for use in future memory extractions.
    
    **Features:**
    - Define custom node types with properties and validation rules
    - Define custom relationship types with constraints
    - Automatic validation against system schemas
    - Support for different scopes (personal, workspace, organization)
    - **Status control**: Set `status` to "active" to immediately activate the schema, or "draft" to save as draft (default)
    - **Enum support**: Use `enum_values` to restrict property values to a predefined list (max 10 values)
    - **Auto-indexing**: Required properties are automatically indexed in Neo4j when schema becomes active
    
    **Schema Limits (optimized for LLM performance):**
    - **Maximum 10 node types** per schema
    - **Maximum 20 relationship types** per schema
    - **Maximum 10 properties** per node type
    - **Maximum 10 enum values** per property
    
    **Property Types & Validation:**
    - `string`: Text values with optional `enum_values`, `min_length`, `max_length`, `pattern`
    - `integer`: Whole numbers with optional `min_value`, `max_value`
    - `float`: Decimal numbers with optional `min_value`, `max_value`
    - `boolean`: True/false values
    - `datetime`: ISO 8601 timestamp strings
    - `array`: Lists of values
    - `object`: Complex nested objects
    
    **Enum Values:**
    - Add `enum_values` to any string property to restrict values to a predefined list
    - Maximum 10 enum values allowed per property
    - Use with `default` to set a default enum value
    - Example: `"enum_values": ["small", "medium", "large"]`
    
    **When to Use Enums:**
    - Limited, well-defined options (≤10 values): sizes, statuses, categories, priorities
    - Controlled vocabularies: "active/inactive", "high/medium/low", "bronze/silver/gold"
    - When you want exact matching and no variations
    
    **When to Avoid Enums:**
    - Open-ended text fields: names, titles, descriptions, addresses
    - Large sets of options (>10): countries, cities, product models
    - When you want semantic similarity matching for entity resolution
    - Dynamic or frequently changing value sets
    
    **Unique Identifiers & Entity Resolution:**
    - Properties marked as `unique_identifiers` are used for entity deduplication and merging
    - **With enum_values**: Exact matching is used - entities with the same enum value are considered identical
    - **Without enum_values**: Semantic similarity matching is used - entities with similar meanings are automatically merged
    - Example: A "name" unique_identifier without enums will merge "Apple Inc" and "Apple Inc." as the same entity
    - Example: A "sku" unique_identifier with enums will only merge entities with exactly matching SKU codes
    - Use enums for unique_identifiers when you have a limited, predefined set of values (≤10 options)
    - Avoid enums for unique_identifiers when you have broad, open-ended values or >10 possible options
    - **Best practices**: Use enums for controlled vocabularies (status codes, categories), avoid for open text (company names, product titles)
    - **In the example above**: "name" uses semantic similarity (open-ended), "sku" uses exact matching (controlled set)
    
    **LLM-Friendly Descriptions:**
    - Write detailed property descriptions that guide the LLM on expected formats and usage
    - Include examples of typical values (e.g., "Product name, typically 2-4 words like 'iPhone 15 Pro'")
    - Specify data formats and constraints clearly (e.g., "Price in USD as decimal number")
    - For enums, explain when to use each option (e.g., "use 'new' for brand new items")
    
    **Authentication Required**:
    One of the following authentication methods must be used:
    - Bearer token in `Authorization` header
    - API Key in `X-API-Key` header
    - Session token in `X-Session-Token` header
    
    **Required Headers**:
    - Content-Type: application/json
    - X-Client-Type: (e.g., 'papr_plugin', 'browser_extension')
    """,
    openapi_extra={
        "operationId": "create_user_schema_v1",
        "x-openai-isConsequential": False
    }
)
async def create_user_schema_v1(
    request: Request,
    schema: UserGraphSchema,
    response: Response,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Depends(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph),
    schema_service: SchemaService = Depends(get_schema_service)
) -> SchemaResponse:
    """Create a new user-defined graph schema"""
    try:
        # Get client type from headers
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')
        logger.info(f"Schema creation - client_type: {client_type}")
        
        # --- Optimized authentication using cached method ---
        auth_start_time = time.time()
        try:
            async with httpx.AsyncClient() as httpx_client:
                if api_key and bearer_token:
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
                    auth_header = request.headers.get('Authorization')
                    auth_response = await get_user_from_token_optimized(auth_header, client_type, memory_graph, httpx_client=httpx_client)
        except ValueError as e:
            logger.error(f"Invalid authentication token: {e}")
            response.status_code = 401
            return SchemaResponse(
                success=False,
                error="Invalid authentication token",
                code=401
            )
        auth_end_time = time.time()
        logger.info(f"Enhanced authentication timing: {(auth_end_time - auth_start_time) * 1000:.2f}ms")
        
        if not auth_response:
            response.status_code = 401
            return SchemaResponse(
                success=False,
                error="Missing or invalid authentication",
                code=401
            )
        
        # Extract user information
        user_id = auth_response.developer_id
        end_user_id = auth_response.end_user_id
        sessionToken = auth_response.session_token
        user_info = auth_response.user_info
        api_key = auth_response.api_key
        workspace_id = auth_response.workspace_id
        
        # Check interaction limits (1 mini interaction for create_user_schema)
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
            
            user_instance = User(id=user_id)
            limit_check = await user_instance.check_interaction_limits_fast(
                interaction_type='mini',
                memory_graph=memory_graph,
                operation=MemoryOperationType.CREATE_SCHEMA,
                api_key_id=api_key_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            if limit_check:
                response_dict, status_code, is_error = limit_check
                if is_error:
                    response.status_code = status_code
                    return SchemaResponse.failure(
                        error=response_dict.get('error'),
                        code=status_code
                    )
        
        logger.info(f"🔍 TRACE STEP 1 - AUTH RESPONSE: user_id={user_id}, end_user_id={end_user_id}")
        logger.info(f"🔍 TRACE STEP 2 - AUTH RESPONSE: workspace_id={workspace_id}")
        logger.info(f"🔍 TRACE STEP 3 - AUTH RESPONSE TYPE: {type(auth_response)}")
        logger.info(f"🔍 TRACE STEP 4 - AUTH RESPONSE ATTRS: {dir(auth_response)}")
        
        # Set user_id and workspace_id if not provided
        logger.info(f"🔍 TRACE STEP 5 - BEFORE SETTING: schema.user_id={schema.user_id}, schema.workspace_id={schema.workspace_id}")
        if not schema.user_id:
            schema.user_id = user_id
            logger.info(f"🔍 TRACE STEP 6 - SET USER_ID: {user_id}")
        if not schema.workspace_id and workspace_id:
            schema.workspace_id = workspace_id
            logger.info(f"🔍 TRACE STEP 7 - SET WORKSPACE_ID: {workspace_id}")
        else:
            logger.warning(f"🔍 TRACE STEP 7 - NOT SETTING WORKSPACE_ID: schema.workspace_id={schema.workspace_id}, workspace_id={workspace_id}")
            
        logger.info(f"🔍 TRACE STEP 8 - AFTER SETTING: schema.user_id={schema.user_id}, schema.workspace_id={schema.workspace_id}")
            
        # Ensure user_id is set (required for schema creation)
        if not schema.user_id:
            response.status_code = 400
            return SchemaResponse(
                success=False,
                error="Unable to determine user_id from authentication",
                code=400
            )
        
        # Extract multi-tenant context from auth response
        organization_id = getattr(auth_response, 'organization_id', None)
        namespace_id = getattr(auth_response, 'namespace_id', None)
        
        # Create schema
        logger.info(f"🔍 TRACE STEP 9 - CALLING SCHEMA SERVICE: user_id={user_id}, workspace_id={workspace_id}, org_id={organization_id}, namespace_id={namespace_id}")
        logger.info(f"🔍 TRACE STEP 10 - SCHEMA OBJECT: user_id={schema.user_id}, workspace_id={schema.workspace_id}")
        result = await schema_service.create_schema(schema, user_id, workspace_id, organization_id, namespace_id)
        
        if not result.success:
            response.status_code = result.code
            return result
        
        # Create Neo4j indexes if schema is ACTIVE
        if result.data and result.data.status == SchemaStatus.ACTIVE:
            logger.info(f"🔧 Schema '{result.data.name}' is ACTIVE, creating Neo4j indexes")
            try:
                # Create indexes in background to avoid blocking the response
                background_tasks.add_task(
                    schema_service.create_indexes_for_schema,
                    result.data,
                    memory_graph
                )
                logger.info(f"✅ Index creation task scheduled for schema '{result.data.name}'")
            except Exception as e:
                # Log error but don't fail the schema creation
                logger.error(f"❌ Failed to schedule index creation for schema '{result.data.name}': {e}")
        else:
            logger.info(f"🔧 Schema '{result.data.name if result.data else 'unknown'}' is not ACTIVE, skipping index creation")
        
        # Log Amplitude event
        background_tasks.add_task(
            _log_amplitude_event_background,
            event_type="create_schema",
            user_info=user_info,
            client_type=client_type,
            amplitude_client=amplitude_client,
            logger=logger,
            api_key=api_key,
            user_id=user_id,
            end_user_id=end_user_id,
            extra_properties={
                'schema_id': result.data.id,
                'schema_name': result.data.name,
                'node_types_count': len(result.data.node_types),
                'relationship_types_count': len(result.data.relationship_types),
                'schema_scope': result.data.scope
            }
        )
        
        response.status_code = 201
        return result
        
    except HTTPException as http_ex:
        logger.error(f"HTTPException in create_user_schema: {http_ex.status_code} - {http_ex.detail}")
        response.status_code = http_ex.status_code
        return SchemaResponse(
            success=False,
            error=http_ex.detail,
            code=http_ex.status_code
        )
    except Exception as e:
        logger.error(f"Error creating schema: {e}", exc_info=True)
        response.status_code = 500
        return SchemaResponse(
            success=False,
            error=str(e),
            code=500
        )

@router.get("",
    response_model=SchemaListResponse,
    responses={
        200: {
            "model": SchemaListResponse,
            "description": "Schemas retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": [
                            {
                                "id": "schema_123",
                                "name": "E-commerce Schema",
                                "description": "Schema for e-commerce operations",
                                "status": "active",
                                "created_at": "2024-01-17T17:30:45.123456Z"
                            }
                        ],
                        "error": None,
                        "code": 200,
                        "total": 1
                    }
                }
            }
        },
        401: {
            "model": SchemaListResponse,
            "description": "Unauthorized"
        },
        500: {
            "model": SchemaListResponse,
            "description": "Internal server error"
        }
    },
    description="""List all schemas accessible to the authenticated user.
    
    Returns schemas that the user owns or has read access to, including:
    - Personal schemas created by the user
    - Workspace schemas shared within the user's workspace
    - Organization schemas available to the user's organization
    
    **Authentication Required**:
    One of the following authentication methods must be used:
    - Bearer token in `Authorization` header
    - API Key in `X-API-Key` header
    - Session token in `X-Session-Token` header
    """,
    openapi_extra={
        "operationId": "list_user_schemas_v1",
        "x-openai-isConsequential": False
    }
)
async def list_user_schemas_v1(
    request: Request,
    response: Response,
    workspace_id: Optional[str] = Query(None, description="Filter by workspace ID"),
    status_filter: Optional[str] = Query(None, description="Filter by status (draft, active, deprecated, archived)"),
    api_key: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Depends(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph),
    schema_service: SchemaService = Depends(get_schema_service)
) -> SchemaListResponse:
    """List all schemas accessible to the authenticated user"""
    try:
        # Get client type from headers
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')
        
        # Authentication
        async with httpx.AsyncClient() as httpx_client:
            if api_key and bearer_token:
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
                auth_header = request.headers.get('Authorization')
                auth_response = await get_user_from_token_optimized(auth_header, client_type, memory_graph, httpx_client=httpx_client)
        
        if not auth_response:
            response.status_code = 401
            return SchemaListResponse(
                success=False,
                error="Missing or invalid authentication",
                code=401
            )
        
        user_id = auth_response.developer_id
        workspace_id = workspace_id or auth_response.workspace_id
        
        # Check interaction limits (1 mini interaction for list_user_schemas)
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
            
            user_instance = User(id=user_id)
            limit_check = await user_instance.check_interaction_limits_fast(
                interaction_type='mini',
                memory_graph=memory_graph,
                operation=MemoryOperationType.LIST_SCHEMAS,
                api_key_id=api_key_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            if limit_check:
                response_dict, status_code, is_error = limit_check
                if is_error:
                    response.status_code = status_code
                    return SchemaListResponse(
                        success=False,
                        data=[],
                        error=response_dict.get('error'),
                        code=status_code,
                        total=0
                    )
        
        # Extract multi-tenant context from auth response
        organization_id = getattr(auth_response, 'organization_id', None)
        logger.info(f"Organization ID: {organization_id}")
        namespace_id = getattr(auth_response, 'namespace_id', None)
        logger.info(f"Namespace ID: {namespace_id}")
        # List schemas
        result = await schema_service.list_schemas(user_id, workspace_id, organization_id, namespace_id)
        
        # Apply status filter if provided
        if result.success and status_filter and result.data:
            filtered_schemas = [
                schema for schema in result.data 
                if schema.status.value == status_filter
            ]
            result.data = filtered_schemas
            result.total = len(filtered_schemas)
        
        response.status_code = result.code
        return result
        
    except Exception as e:
        logger.error(f"Error listing schemas: {e}", exc_info=True)
        response.status_code = 500
        return SchemaListResponse(
            success=False,
            error=str(e),
            code=500
        )

@router.get("/{schema_id}",
    response_model=SchemaResponse,
    responses={
        200: {"model": SchemaResponse, "description": "Schema retrieved successfully"},
        401: {"model": SchemaResponse, "description": "Unauthorized"},
        404: {"model": SchemaResponse, "description": "Schema not found"},
        500: {"model": SchemaResponse, "description": "Internal server error"}
    },
    description="""Get a specific schema by ID.
    
    Returns the complete schema definition including node types, relationship types,
    and metadata. User must have read access to the schema.
    """,
    openapi_extra={
        "operationId": "get_user_schema_v1",
        "x-openai-isConsequential": False
    }
)
async def get_user_schema_v1(
    schema_id: str,
    request: Request,
    response: Response,
    api_key: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Depends(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph),
    schema_service: SchemaService = Depends(get_schema_service)
) -> SchemaResponse:
    """Get a specific schema by ID"""
    try:
        # Get client type from headers
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')
        
        # Authentication
        async with httpx.AsyncClient() as httpx_client:
            if api_key and bearer_token:
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
                auth_header = request.headers.get('Authorization')
                auth_response = await get_user_from_token_optimized(auth_header, client_type, memory_graph, httpx_client=httpx_client)
        
        if not auth_response:
            response.status_code = 401
            return SchemaResponse(
                success=False,
                error="Missing or invalid authentication",
                code=401
            )
        
        user_id = auth_response.developer_id
        
        # Check interaction limits (1 mini interaction for get_user_schema)
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
            
            user_instance = User(id=user_id)
            limit_check = await user_instance.check_interaction_limits_fast(
                interaction_type='mini',
                memory_graph=memory_graph,
                operation=MemoryOperationType.GET_SCHEMA,
                api_key_id=api_key_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            if limit_check:
                response_dict, status_code, is_error = limit_check
                if is_error:
                    response.status_code = status_code
                    return SchemaResponse.failure(
                        error=response_dict.get('error'),
                        code=status_code
                    )
        
        # Extract multi-tenant context from auth response
        organization_id = getattr(auth_response, 'organization_id', None)
        namespace_id = getattr(auth_response, 'namespace_id', None)
        
        # Get schema
        result = await schema_service.get_schema(schema_id, user_id, None, organization_id, namespace_id)
        
        response.status_code = result.code
        return result
        
    except Exception as e:
        logger.error(f"Error getting schema: {e}", exc_info=True)
        response.status_code = 500
        return SchemaResponse(
            success=False,
            error=str(e),
            code=500
        )

@router.put("/{schema_id}",
    response_model=SchemaResponse,
    responses={
        200: {"model": SchemaResponse, "description": "Schema updated successfully"},
        401: {"model": SchemaResponse, "description": "Unauthorized"},
        403: {"model": SchemaResponse, "description": "Insufficient permissions"},
        404: {"model": SchemaResponse, "description": "Schema not found"},
        500: {"model": SchemaResponse, "description": "Internal server error"}
    },
    description="""Update an existing schema.
    
    Allows modification of schema properties, node types, relationship types, and status.
    User must have write access to the schema. Updates create a new version
    while preserving the existing data.
    
    **Status Management:**
    - Set `status` to "active" to activate the schema and trigger Neo4j index creation
    - Set `status` to "draft" to deactivate the schema
    - Set `status` to "archived" to soft-delete the schema
    """,
    openapi_extra={
        "operationId": "update_user_schema_v1",
        "x-openai-isConsequential": False
    }
)
async def update_user_schema_v1(
    schema_id: str,
    updates: Dict[str, Any],
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Depends(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph),
    schema_service: SchemaService = Depends(get_schema_service)
) -> SchemaResponse:
    """Update an existing schema"""
    try:
        # Get client type from headers
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')
        
        # Authentication
        async with httpx.AsyncClient() as httpx_client:
            if api_key and bearer_token:
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
                auth_header = request.headers.get('Authorization')
                auth_response = await get_user_from_token_optimized(auth_header, client_type, memory_graph, httpx_client=httpx_client)
        
        if not auth_response:
            response.status_code = 401
            return SchemaResponse(
                success=False,
                error="Missing or invalid authentication",
                code=401
            )
        
        user_id = auth_response.developer_id
        user_info = auth_response.user_info
        end_user_id = auth_response.end_user_id
        
        # Check interaction limits (1 mini interaction for update_user_schema)
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
            
            user_instance = User(id=user_id)
            limit_check = await user_instance.check_interaction_limits_fast(
                interaction_type='mini',
                memory_graph=memory_graph,
                operation=MemoryOperationType.UPDATE_SCHEMA,
                api_key_id=api_key_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            if limit_check:
                response_dict, status_code, is_error = limit_check
                if is_error:
                    response.status_code = status_code
                    return SchemaResponse.failure(
                        error=response_dict.get('error'),
                        code=status_code
                    )
        
        # Update schema
        result = await schema_service.update_schema(schema_id, updates, user_id)
        
        if result.success:
            # Create Neo4j indexes if schema status was changed to ACTIVE
            if 'status' in updates and updates['status'] == SchemaStatus.ACTIVE.value and result.data:
                logger.info(f"🔧 Schema '{result.data.name}' status changed to ACTIVE, creating Neo4j indexes")
                try:
                    # Create indexes in background to avoid blocking the response
                    background_tasks.add_task(
                        schema_service.create_indexes_for_schema,
                        result.data,
                        memory_graph
                    )
                    logger.info(f"✅ Index creation task scheduled for schema '{result.data.name}'")
                except Exception as e:
                    # Log error but don't fail the schema update
                    logger.error(f"❌ Failed to schedule index creation for schema '{result.data.name}': {e}")
            
            # Log Amplitude event
            event_type = "activate_schema" if ('status' in updates and updates['status'] == SchemaStatus.ACTIVE.value) else "update_schema"
            background_tasks.add_task(
                _log_amplitude_event_background,
                event_type=event_type,
                user_info=user_info,
                client_type=client_type,
                amplitude_client=amplitude_client,
                logger=logger,
                api_key=api_key,
                user_id=user_id,
                end_user_id=end_user_id,
                extra_properties={
                    'schema_id': schema_id,
                    'updated_fields': list(updates.keys())
                }
            )
        
        response.status_code = result.code
        return result
        
    except Exception as e:
        logger.error(f"Error updating schema: {e}", exc_info=True)
        response.status_code = 500
        return SchemaResponse(
            success=False,
            error=str(e),
            code=500
        )

@router.delete("/{schema_id}",
    responses={
        200: {"description": "Schema deleted successfully"},
        401: {"description": "Unauthorized"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Schema not found"},
        500: {"description": "Internal server error"}
    },
    description="""Delete a schema.
    
    Soft deletes the schema by marking it as archived. The schema data and
    associated graph nodes/relationships are preserved for data integrity.
    User must have write access to the schema.
    """,
    openapi_extra={
        "operationId": "delete_user_schema_v1",
        "x-openai-isConsequential": False
    }
)
async def  delete_user_schema_v1(
    schema_id: str,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Depends(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph),
    schema_service: SchemaService = Depends(get_schema_service)
):
    """Delete a schema"""
    try:
        # Get client type from headers
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')
        
        # Authentication
        async with httpx.AsyncClient() as httpx_client:
            if api_key and bearer_token:
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
                auth_header = request.headers.get('Authorization')
                auth_response = await get_user_from_token_optimized(auth_header, client_type, memory_graph, httpx_client=httpx_client)
        
        if not auth_response:
            response.status_code = 401
            return {"error": "Missing or invalid authentication"}
        
        user_id = auth_response.developer_id
        user_info = auth_response.user_info
        end_user_id = auth_response.end_user_id
        
        # Check interaction limits (1 mini interaction for delete_user_schema)
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
            
            user_instance = User(id=user_id)
            limit_check = await user_instance.check_interaction_limits_fast(
                interaction_type='mini',
                memory_graph=memory_graph,
                operation=MemoryOperationType.DELETE_SCHEMA,
                api_key_id=api_key_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            if limit_check:
                response_dict, status_code, is_error = limit_check
                if is_error:
                    response.status_code = status_code
                    return {"error": response_dict.get('error')}
        
        # Delete schema
        result = await schema_service.delete_schema(schema_id, user_id)
        
        if not result.success:
            response.status_code = result.code
            return {"error": result.error}
        
        # Log Amplitude event
        background_tasks.add_task(
            _log_amplitude_event_background,
            event_type="delete_schema",
            user_info=user_info,
            client_type=client_type,
            amplitude_client=amplitude_client,
            logger=logger,
            api_key=api_key,
            user_id=user_id,
            end_user_id=end_user_id,
            extra_properties={
                'schema_id': schema_id
            }
        )
        
        response.status_code = 200
        return {"message": "Schema deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting schema: {e}", exc_info=True)
        response.status_code = 500
        return {"error": str(e)}


# Helper functions
async def _log_amplitude_event_background(
    event_type: str,
    user_info: Optional[Dict[str, Any]],
    client_type: str,
    amplitude_client: Amplitude,
    logger,
    api_key: Optional[str],
    user_id: str,
    end_user_id: str,
    extra_properties: Optional[Dict[str, Any]] = None
):
    """Background task to log Amplitude events"""
    try:
        success = await log_amplitude_event(
            event_type=event_type,
            user_info=user_info,
            client_type=client_type,
            amplitude_client=amplitude_client,
            logger=logger,
            extra_properties=extra_properties or {},
            end_user_id=end_user_id
        )
        if success:
            logger.info(f"Amplitude event logged successfully for {event_type}")
        else:
            logger.warning(f"Failed to log Amplitude event for {event_type}")
    except Exception as e:
        logger.error(f"Error logging event to Amplitude: {e}")

