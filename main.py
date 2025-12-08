from fastapi import FastAPI, HTTPException, Request, Depends, Response, Form, File, UploadFile, BackgroundTasks, Header, Query, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse, HTMLResponse
from typing import Dict, Any, Optional, List, Union
from services.auth_utils import get_user_from_token, get_user_from_token_optimized, get_oauth_client, determine_client_type, CustomGetToken
from routes.memory_routes import common_add_memory_handler, common_add_memory_batch_handler
import magic
import os
import sys
from werkzeug.utils import secure_filename
from services.user_utils import User
from memory.memory_graph import MemoryGraph
from amplitude import Amplitude, BaseEvent
import json
from dotenv import find_dotenv, load_dotenv
from os import environ as env
from urllib.parse import quote_plus, urlencode
from authlib.integrations.starlette_client import OAuth
import secrets
from fastapi.staticfiles import StaticFiles
import aiofiles
from starlette.middleware.sessions import SessionMiddleware
from starlette.config import Config
from contextlib import contextmanager, asynccontextmanager
from werkzeug.local import LocalProxy
from starlette.datastructures import FormData, UploadFile as StarletteUploadFile
from werkzeug.datastructures import MultiDict
from models.parse_server import (
    ParseStoredMemory, AddMemoryResponse, ErrorDetail, DeletionStatus, BatchMemoryResponse, BatchMemoryError, DeleteMemoryResponse, UpdateMemoryResponse, UpdateMemoryItem, SystemUpdateStatus, DocumentUploadResponse, DocumentUploadStatus, AddMemoryItem, Memory
)
from models.memory_models import GetMemoryResponse, SearchResponse, SearchRequest, SearchResult, AddMemoryRequest, BatchMemoryRequest, UpdateMemoryRequest, MemoryMetadata, RerankingConfig
from api_handlers.chat_gpt_completion import ChatGPTCompletion
from amplitude import Identify, BaseEvent, EventOptions
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from services.logging_config import get_logger
from fastapi.security import APIKeyHeader, OAuth2AuthorizationCodeBearer
import asyncio
import certifi
from fastapi.middleware.gzip import GZipMiddleware
import uuid
from services.memory_management import get_document_upload_status, retrieve_memory_item_by_qdrant_id
from models.parse_server import DocumentUploadStatus, DocumentUploadStatusType
from models.shared_types import UploadDocumentRequest, LoginResponse, TokenRequest, TokenResponse, UserInfoResponse, LogoutResponse, CallbackResponse, ErrorResponse
from models.user_schemas import UserGraphSchema, SchemaResponse, SchemaListResponse
from services.schema_service import SchemaService, get_schema_service
from routers.v1.schema_routes_v1 import router as schema_router_v1

import fastapi  # Add this import for exception handling
from fastapi.encoders import jsonable_encoder
import httpx
from services.logger_singleton import LoggerSingleton
import secrets
import base64
from azure.monitor.opentelemetry import configure_azure_monitor
from fastapi import APIRouter
from fastapi import Depends, Security
from fastapi.security import HTTPBearer, APIKeyHeader, HTTPAuthorizationCredentials, OAuth2AuthorizationCodeBearer
from fastapi.openapi.utils import get_openapi
import yaml
from models.memory_models import MemoryMetadata
from collections import defaultdict
from app_factory import create_app
from services.utils import log_amplitude_event, serialize_datetime, get_memory_graph
from services.error_handlers import register_exception_handlers

# Load environment variables
# Skip loading .env file if USE_DOTENV is set to false (for shell-based env var setup)
USE_DOTENV = env.get("USE_DOTENV", "true").lower() == "true"
if USE_DOTENV:
    # Load .env first (base configuration)
    ENV_FILE = find_dotenv()
    if ENV_FILE:
        load_dotenv(ENV_FILE)
    # Load .env.local second (local overrides with override=True)
    load_dotenv(".env.local", override=True)

# Add this near the top with your other env variables
HIDE_INTERNAL_ENDPOINTS = env.get("HIDE_INTERNAL_ENDPOINTS", "True").lower() == "true"

logger = LoggerSingleton.get_logger(__name__)
logger.info("Logger initialized at top of main.py!")


app = create_app()

# Register database exception handlers
register_exception_handlers(app)

oauth = app.state.oauth
bearer_auth = app.state.bearer_auth
api_key_header = app.state.api_key_header
session_token_header = app.state.session_token_header
# Use generic telemetry service instead of direct amplitude_client
from core.services.telemetry import get_telemetry
telemetry = get_telemetry()
# Keep amplitude_client for backward compatibility in routes that still reference it
amplitude_client = None  # Will be deprecated in favor of telemetry service
chat_gpt = app.state.chat_gpt



auth_code_locks = {}
auth0_domain = env.get("AUTH0_DOMAIN")
public_server_url = env.get("PARSE_SERVER_URL")
web_app_url = env.get("WEB_APP_URL")
api_key = env.get("OPENAI_API_KEY")
organization_id = env.get("OPENAI_ORGANIZATION")

PARSE_SERVER_URL = env.get("PARSE_SERVER_URL") 
logger.info(f"PARSE_SERVER_URL: {PARSE_SERVER_URL}")

MEMORY_SERVER_URL = env.get("PYTHON_SERVER_URL")
logger.info(f"MEMORY_SERVER_URL: {MEMORY_SERVER_URL}")

# Log MongoDB connection info
MONGO_URI = env.get("MONGO_URI")
DATABASE_URI = env.get("DATABASE_URI")

if MONGO_URI:
    # Mask sensitive parts of the URI for logging
    masked_mongo_uri = MONGO_URI
    if '@' in masked_mongo_uri:
        # Extract just the host and database parts for logging
        parts = masked_mongo_uri.split('@')
        if len(parts) > 1:
            host_db_part = parts[1]
            logger.info(f"MONGO_URI host/db: {host_db_part}")
        else:
            logger.info("MONGO_URI: [format not recognized]")
    else:
        logger.info("MONGO_URI: [no auth info]")
elif DATABASE_URI and "mongodb" in DATABASE_URI:
    # MONGO_URI will be derived from DATABASE_URI at runtime (same database as Parse Server)
    logger.info("MONGO_URI: NOT SET (will use same database as Parse Server)")
else:
    logger.info("MONGO_URI: NOT SET") 
HEADERS = {
    "X-Parse-Application-Id": env.get("PARSE_APPLICATION_ID"),
    "X-Parse-REST-API-Key": env.get("PARSE_REST_API_KEY"),
    "Content-Type": "application/json"
}

NEO4J_URL = env.get("NEO4J_URL")

# Add prefix to keys to avoid conflicts with Celery tasks
REDIS_KEY_PREFIX = env.get('CELERY_REDIS_KEY_PREFIX', 'doc-upload')

# Initialize token handler
get_token = CustomGetToken(env.get("AUTH0_DOMAIN"))


class RequestContextAdapter:
    def __init__(self, fastapi_request: Request):
        self._request = fastapi_request
        self._form_data = None
        self._json_data = None
        self._headers = None

    @property
    def headers(self):
        if self._headers is None:
            self._headers = dict(self._request.headers)
        return self._headers

    async def get_form(self):
        if self._form_data is None:
            self._form_data = await self._request.form()
            if isinstance(self._form_data, FormData):
                self._form_data = MultiDict(self._form_data._dict)
        return self._form_data

    async def get_files(self):
        if self._form_data is None:
            self._form_data = await self._request.form()
        return {k: v for k, v in self._form_data.items() if isinstance(v, StarletteUploadFile)}

    async def get_json(self):
        if self._json_data is None:
            self._json_data = await self._request.json()
        return self._json_data

    # Flask-like properties
    @property
    def form(self):
        return self._form_data

    @property
    def files(self):
        return self._files

    def get_header(self, key: str, default: str = None) -> str:
        return self.headers.get(key, default)

@asynccontextmanager
async def request_context(fastapi_request: Request):
    """Context manager to provide Flask-like request context for FastAPI requests."""
    adapter = RequestContextAdapter(fastapi_request)
    
    # Pre-load form data and files if needed
    content_type = adapter.headers.get('Content-Type', '')
    if 'multipart/form-data' in content_type or 'application/x-www-form-urlencoded' in content_type:
        adapter._form_data = await adapter.get_form()
        adapter._files = await adapter.get_files()
    elif 'application/json' in content_type:
        adapter._json_data = await adapter.get_json()
    
    # Store the adapter in a way that's accessible to the Flask code
    import flask
    flask.request = adapter
    try:
        yield adapter
    finally:
        # Clean up
        flask.request = None

# Add global error handler
@app.middleware("http")
async def global_error_handler(request: Request, call_next):
    try:
        return await call_next(request)
    except fastapi.exceptions.ResponseValidationError as e:
        logger.error(f"Response validation error for {request.url.path}: {e}", exc_info=True)
        
        try:
            # Handle different endpoints with their appropriate response models
            if "/add_memory" in request.url.path:
                return JSONResponse(
                    status_code=500,
                    content=AddMemoryResponse.failure(
                        error="Internal server error",
                        code=500
                    ).model_dump()
                )
            elif '/v1/memory/batch' in request.url.path:
                return JSONResponse(
                    status_code=500,
                    content=BatchMemoryResponse.failure(
                        errors=[],
                        code=500,
                        error="Internal server error"
                    ).model_dump()
                )
            elif '/v1/memory' in request.url.path:
                # Check the HTTP method to determine the correct response model
                if request.method == "POST":
                    return JSONResponse(
                        status_code=500,
                        content=AddMemoryResponse.failure(
                            error="Internal server error",
                            code=500
                        ).model_dump()
                    )
                elif request.method == "PUT":
                    return JSONResponse(
                        status_code=500,
                        content=UpdateMemoryResponse.failure(
                            error="Internal server error",
                            code=500
                        ).model_dump()
                    )
                elif request.method == "DELETE":
                    return JSONResponse(
                        status_code=500,
                        content=DeleteMemoryResponse.failure(
                            error="Internal server error",
                            code=500
                        ).model_dump()
                    )
                elif request.method == "GET":
                    return JSONResponse(
                        status_code=500,
                        content=SearchResponse.failure(
                            error="Internal server error",
                            code=500
                        ).model_dump()
                    )
            elif "/get_memory" in request.url.path:
                return JSONResponse(
                    status_code=500,
                    content=SearchResponse.failure(
                        error=str(e),
                        code=500
                    ).model_dump()
                )
            elif "/v1/search" in request.url.path:
                return JSONResponse(
                    status_code=500,
                    content=SearchResponse.failure(
                        error=str(e),
                        code=500
                    ).model_dump()
                )
            elif "/delete_memory" in request.url.path:
                return JSONResponse(
                    status_code=500,
                    content=DeleteMemoryResponse.failure(
                        error=str(e),
                        code=500
                    ).model_dump()
                )
            elif "/update_memory" in request.url.path:
                return JSONResponse(
                    status_code=500,
                    content=UpdateMemoryResponse.failure(
                        error=str(e),
                        code=500
                    ).model_dump()
                )
            elif "/add_document" in request.url.path:
                return JSONResponse(
                    status_code=500,
                    content=DocumentUploadResponse.failure(
                        error="Error processing document",
                        code=500
                    ).model_dump()
                )
            else:
                return JSONResponse(
                    status_code=500,
                    content=ErrorDetail(
                        code=500,
                        detail=str(e)
                    ).model_dump()
                )
                
        except Exception as validation_error:
            # If we fail to create the error response, fall back to a simple error
            logger.error(f"Error creating error response: {validation_error}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"code": 500, "detail": "Internal server error during error handling"}
            )
            
    except HTTPException as http_ex:
        # Handle HTTPExceptions by converting them to appropriate JSON responses
        logger.error(f"HTTPException for {request.url.path}: {http_ex.status_code} - {http_ex.detail}")
        
        try:
            # Handle different endpoints with their appropriate response models
            if "/add_memory" in request.url.path:
                return JSONResponse(
                    status_code=http_ex.status_code,
                    content=AddMemoryResponse.failure(
                        error=http_ex.detail,
                        code=http_ex.status_code
                    ).model_dump()
                )
            elif '/v1/memory/batch' in request.url.path:
                return JSONResponse(
                    status_code=http_ex.status_code,
                    content=BatchMemoryResponse.failure(
                        errors=[],
                        code=http_ex.status_code,
                        error=http_ex.detail
                    ).model_dump()
                )
            elif '/v1/memory' in request.url.path:
                # Check the HTTP method to determine the correct response model
                if request.method == "POST":
                    return JSONResponse(
                        status_code=http_ex.status_code,
                        content=AddMemoryResponse.failure(
                            error=http_ex.detail,
                            code=http_ex.status_code
                        ).model_dump()
                    )
                elif request.method == "PUT":
                    return JSONResponse(
                        status_code=http_ex.status_code,
                        content=UpdateMemoryResponse.failure(
                            error=http_ex.detail,
                            code=http_ex.status_code
                        ).model_dump()
                    )
                elif request.method == "DELETE":
                    return JSONResponse(
                        status_code=http_ex.status_code,
                        content=DeleteMemoryResponse.failure(
                            error=http_ex.detail,
                            code=http_ex.status_code
                        ).model_dump()
                    )
                elif request.method == "GET":
                    return JSONResponse(
                        status_code=http_ex.status_code,
                        content=SearchResponse.failure(
                            error=http_ex.detail,
                            code=http_ex.status_code
                        ).model_dump()
                    )
            elif "/get_memory" in request.url.path:
                return JSONResponse(
                    status_code=http_ex.status_code,
                    content=SearchResponse.failure(
                        error=http_ex.detail,
                        code=http_ex.status_code
                    ).model_dump()
                )
            elif "/v1/search" in request.url.path:
                return JSONResponse(
                    status_code=http_ex.status_code,
                    content=SearchResponse.failure(
                        error=http_ex.detail,
                        code=http_ex.status_code
                    ).model_dump()
                )
            elif "/delete_memory" in request.url.path:
                return JSONResponse(
                    status_code=http_ex.status_code,
                    content=DeleteMemoryResponse.failure(
                        error=http_ex.detail,
                        code=http_ex.status_code
                    ).model_dump()
                )
            elif "/update_memory" in request.url.path:
                return JSONResponse(
                    status_code=http_ex.status_code,
                    content=UpdateMemoryResponse.failure(
                        error=http_ex.detail,
                        code=http_ex.status_code
                    ).model_dump()
                )
            elif "/add_document" in request.url.path:
                return JSONResponse(
                    status_code=http_ex.status_code,
                    content=DocumentUploadResponse.failure(
                        error=http_ex.detail,
                        code=http_ex.status_code
                    ).model_dump()
                )
            else:
                return JSONResponse(
                    status_code=http_ex.status_code,
                    content=ErrorDetail(
                        code=http_ex.status_code,
                        detail=http_ex.detail
                    ).model_dump()
                )
                
        except Exception as validation_error:
            # If we fail to create the error response, fall back to a simple error
            logger.error(f"Error creating HTTPException response: {validation_error}", exc_info=True)
            return JSONResponse(
                status_code=http_ex.status_code,
                content={"code": http_ex.status_code, "detail": http_ex.detail}
            )
    except Exception as e:
        logger.error(f"Unhandled error for {request.url.path}: {e}", exc_info=True)
        
        try:
            # Similar pattern for unhandled exceptions
            if "/add_memory" in request.url.path:
                return JSONResponse(
                    status_code=500,
                    content=AddMemoryResponse.failure(
                        error="Internal server error",
                        code=500
                    ).model_dump()
                )
            elif "/get_memory" in request.url.path:
                return JSONResponse(
                    status_code=500,
                    content=SearchResponse.failure(
                        error=f"Internal server error: {str(e)}",
                        code=500
                    ).model_dump()
                )
            elif "/delete_memory" in request.url.path:
                return JSONResponse(
                    status_code=500,
                    content=DeleteMemoryResponse.failure(
                        error=f"Internal server error: {str(e)}",
                        code=500
                    ).model_dump()
                )
            elif "/update_memory" in request.url.path:
                return JSONResponse(
                    status_code=500,
                    content=UpdateMemoryResponse.failure(
                        error=f"Internal server error: {str(e)}",
                        code=500
                    ).model_dump()
                )
            elif "/add_document" in request.url.path:
                return JSONResponse(
                    status_code=500,
                    content=DocumentUploadResponse.failure(
                        error="Error processing document",
                        code=500
                    ).model_dump()
                )
            else:
                return JSONResponse(
                    status_code=500,
                    content=ErrorDetail(
                        code=500,
                        detail=f"Internal server error: {str(e)}"
                    ).model_dump()
                )
                
        except Exception as response_error:
            # If we fail to create the error response, fall back to a simple error
            logger.error(f"Error creating error response: {response_error}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"code": 500, "detail": "Internal server error during error handling"}
            )


@app.get("/restart", include_in_schema=not HIDE_INTERNAL_ENDPOINTS)
async def restart_server():
    """Endpoint to restart the FastAPI application"""
    logger.info("Manual restart requested")
    # Flush output buffers
    sys.stdout.flush()
    sys.stderr.flush()
    # Execute the current process again
    os.execv(sys.executable, [sys.executable] + sys.argv)

@app.get("/", include_in_schema=not HIDE_INTERNAL_ENDPOINTS)
async def home():
    return {"message": "Welcome to my Papr Memory API!"}

@app.get("/.well-known/ai-plugin.json", include_in_schema=not HIDE_INTERNAL_ENDPOINTS)
@app.get("/ai-plugin.json", include_in_schema=not HIDE_INTERNAL_ENDPOINTS)
async def serve_ai_plugin():
    # Get server URL from environment variable
    server_url = env.get("PYTHON_SERVER_URL", "https://memory.papr.ai")
    
    # Create ai-plugin.json dynamically
    ai_plugin_data = {
        "schema_version": "v1",
        "name_for_model": "Papr_Memory",
        "name_for_human": "Papr Memory",
        "description_for_model": "This plugin allows the model to access and interact with the user's personal memory graph. It can retrieve information from past conversations, meeting transcripts, documents, and more. Use it when the user's query pertains to information that may be stored in their personal memory.",
        "description_for_human": "Retrieve information from your personal memory includes past conversations.",
        "auth": {
            "type": "oauth",
            "scope": "openid profile email offline_access",
            "authorization_url": f"{server_url}/login",
            "token_url": f"{server_url}/token",
            "authorization_content_type": "application/x-www-form-urlencoded",
            "verification_tokens": {
                "openai": "58c0b53d09a04516aa45f4d34bb3e3db"
            }
        },
        "api": {
            "type": "openapi",
            "url": f"{server_url}/openapi.yaml"
        },
        "logo_url": f"{server_url}/logo.png",
        "contact_email": "shawkat@papr.ai",
        "legal_info_url": "https://www.papr.ai/terms-of-service"
    }
    
    return JSONResponse(content=ai_plugin_data)

@app.get("/.well-known/openapi.yaml", include_in_schema=not HIDE_INTERNAL_ENDPOINTS)
@app.get("/openapi.yaml", include_in_schema=not HIDE_INTERNAL_ENDPOINTS)
async def serve_openapi():
    # Get server URL from environment variable
    server_url = env.get("PYTHON_SERVER_URL", "https://memory.papr.ai")
    
    # Generate OpenAPI schema dynamically
    openapi_schema = get_openapi(
        title="Papr Memory API",
        version="1.0.0",
        description="API for managing personal memory graph with OAuth2 authentication",
        routes=app.routes,
        servers=[{"url": server_url}]
    )
    
    # Convert to YAML
    yaml_content = yaml.dump(openapi_schema, default_flow_style=False, sort_keys=False)
    
    return Response(content=yaml_content, media_type='text/yaml')

@app.get("/.well-known/logo.png", include_in_schema=not HIDE_INTERNAL_ENDPOINTS)
@app.get("/logo.png", include_in_schema=not HIDE_INTERNAL_ENDPOINTS)
async def serve_logo():
    return FileResponse('.well-known/logo.png', media_type='image/png')

@app.get("/health", include_in_schema=not HIDE_INTERNAL_ENDPOINTS)
async def health_check():
    """Basic health check that returns healthy if the application is running"""
    return {
        "status": "healthy",
        "message": "Service is running"
    }, status.HTTP_200_OK

@app.get("/login",
    response_model=LoginResponse,
    responses={
        200: {"model": LoginResponse, "description": "OAuth2 login initiated successfully"},
        400: {"model": ErrorResponse, "description": "Bad request - missing redirect URI"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    description="""OAuth2 login endpoint. Initiates the OAuth2 authorization code flow.
    
    **Query Parameters:**
    - `redirect_uri`: The URI to redirect to after authentication (required)
    - `state`: A random string for CSRF protection (optional but recommended)
    
    **Flow:**
    1. Client redirects user to this endpoint with `redirect_uri` and `state`
    2. This endpoint redirects user to Auth0 for authentication
    3. After authentication, Auth0 redirects to `/callback` with authorization code
    4. `/callback` redirects back to the original `redirect_uri` with code and state
    
    **Example:**
    ```
    GET /login?redirect_uri=https://chat.openai.com&state=abc123
    ```
    """,
    openapi_extra={
        "operationId": "oauth2_login",
        "tags": ["Authentication"]
    }
)
async def login(request: Request):
    auth_header = request.headers.get('Authorization')
    state = request.query_params.get('state')
    redirect_uri = request.query_params.get('redirect_uri')

    logger.info(f"[LOGIN] Received login request: redirect_uri={redirect_uri}, state={state}, headers={dict(request.headers)}")

    if not redirect_uri:
        logger.error("[LOGIN] Missing redirect_uri in request")
        raise HTTPException(status_code=400, detail="Missing redirect URI")

    client_type = determine_client_type(redirect_uri)
    auth0 = get_oauth_client(oauth, client_type)
    current_time = datetime.now(timezone.utc)

    # Store auth state data
    request.session[f'auth_state_{state}'] = {
        'redirect_uri': redirect_uri,
        'client_type': client_type,
        'created_at': current_time.isoformat(),
        'expires': (current_time + timedelta(minutes=10)).isoformat()
    }
    logger.info(f"[LOGIN] Stored state in session: key=auth_state_{state}, value={request.session[f'auth_state_{state}']}")

    logger.info(f"[LOGIN] Redirecting to Auth0 with state={state} and redirect_uri={redirect_uri}")
    return await auth0.authorize_redirect(
        request,
        redirect_uri=redirect_uri,
        state=state,
        scope="openid profile email offline_access"
    )

@app.get("/callback", 
    response_model=CallbackResponse,
    responses={
        200: {"model": CallbackResponse, "description": "OAuth2 callback processed successfully"},
        400: {"model": ErrorResponse, "description": "Bad request - missing code or state"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    description="""OAuth2 callback endpoint. Processes the authorization code from Auth0.
    
    **Query Parameters:**
    - `code`: Authorization code from Auth0 (required)
    - `state`: State parameter for CSRF protection (required)
    
    **Flow:**
    1. Auth0 redirects to this endpoint after successful authentication
    2. This endpoint validates the authorization code and state
    3. Redirects back to the original `redirect_uri` with code and state
    4. Client can then exchange the code for tokens at `/token` endpoint
    
    **Security:**
    - Validates state parameter to prevent CSRF attacks
    - Checks authorization code expiration
    - Cleans up session data after processing
    """,
    openapi_extra={
        "operationId": "oauth2_callback",
        "tags": ["Authentication"]
    }
)
async def callback(request: Request):
    logger.info("[CALLBACK] Starting callback processing")
    try:
        auth_code = request.query_params.get('code')
        encoded_state = request.query_params.get('state')
        logger.info(f"[CALLBACK] Received: code={auth_code}, state={encoded_state}")

        if not auth_code or not encoded_state:
            logger.error("[CALLBACK] Missing code or state in callback")
            raise HTTPException(status_code=400, detail="Missing authorization code or state")

        auth_state = request.session.get(f'auth_state_{encoded_state}')
        logger.info(f"[CALLBACK] Fetched auth_state from session: {auth_state}")

        if not auth_state:
            logger.error(f"[CALLBACK] No auth state found for state: {encoded_state}")
            raise HTTPException(status_code=400, detail="Invalid or expired state")

        # Check expiration
        if datetime.fromisoformat(auth_state['expires']) < datetime.now(timezone.utc):
            logger.error(f"[CALLBACK] Auth state expired for state: {encoded_state}")
            raise HTTPException(status_code=400, detail="State expired")

        redirect_uri = auth_state.get('redirect_uri')
        client_type = auth_state.get('client_type')
        logger.info(f"[CALLBACK] Using redirect_uri={redirect_uri}, client_type={client_type}")

        # Store auth code data
        request.session[f'auth_code_{auth_code}'] = {
            'state': encoded_state,
            'client_type': client_type,
            'redirect_uri': redirect_uri,
            'expires': (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        }
        logger.info(f"[CALLBACK] Stored auth_code in session: key=auth_code_{auth_code}")

        # Clean up the auth state data
        del request.session[f'auth_state_{encoded_state}']
        logger.info(f"[CALLBACK] Cleaned up auth_state for state: {encoded_state}")

        # Redirect back to the original redirect URI with auth code and state
        redirect_url = f"{redirect_uri}?code={auth_code}&state={encoded_state}"
        logger.info(f"[CALLBACK] Redirecting user to: {redirect_url}")
        return RedirectResponse(url=redirect_url, status_code=302)

    except Exception as e:
        logger.error(f"[CALLBACK] Error during callback: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/complete-auth")
async def complete_auth(
    return_token: str,
    request: Request
):
    try:
        # Get stored return information
        return_info = request.session.get(f'return_info_{return_token}')
        if not return_info:
            raise HTTPException(status_code=400, detail="Invalid or expired token")

        # Check expiration (convert stored ISO string back to datetime)
        if datetime.fromisoformat(return_info['expires']) < datetime.now(timezone.utc):
            raise HTTPException(status_code=400, detail="Token expired")

        # Clean up session
        del request.session[f'return_info_{return_token}']

        # Build redirect URL for ChatGPT with original code and state
        redirect_url = f"{return_info['redirect_uri']}?code={return_info['auth_code']}&state={return_info['state']}"
        
        # Log the redirect for debugging
        logger.info(f"Redirecting to ChatGPT: {redirect_url}")
        
        # Return a redirect response to send the user back to ChatGPT
        return RedirectResponse(url=redirect_url, status_code=302)

    except Exception as e:
        logger.error(f"Error completing auth: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# Helper function to check if user needs onboarding
async def check_if_user_needs_onboarding(request: Request, auth_code: str, client_type: str) -> bool:
    try:
        auth0 = get_oauth_client(oauth, client_type)
        logger.info(f"Got Auth0 client for {client_type}")
        
        try:
            token = await auth0.authorize_access_token(request)
            logger.info("Successfully got access token")
            
            access_token = token.get('access_token')
            if not access_token:
                logger.error("No access token in response")
                return True
                
            user_info = await User.verify_access_token(access_token, client_type)
            if not user_info:
                logger.error("Could not verify access token")
                return True
                
            logger.info(f"User info from Auth0: {user_info}")
            
            # Get completedProfileSignup from the correct path
            completed_profile_signup = user_info.get('https://papr.scope.com/completedProfileSignup', False)
            logger.info(f"Completed profile signup: {completed_profile_signup}")
            
            return not completed_profile_signup
            
        except Exception as e:
            logger.error(f"Error during Auth0 token exchange: {str(e)}")
            return True
            
    except Exception as e:
        logger.error(f"Error checking onboarding status: {str(e)}")
        return True

@app.post("/token", 
    response_model=TokenResponse,
    responses={
        200: {"model": TokenResponse, "description": "OAuth2 token exchange successful"},
        400: {"model": ErrorResponse, "description": "Bad request - invalid grant type or missing parameters"},
        401: {"model": ErrorResponse, "description": "Unauthorized - invalid authorization code"},
        415: {"model": ErrorResponse, "description": "Unsupported Media Type"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    description="""OAuth2 token endpoint. Exchanges authorization code for access tokens.
    
    **Request Body (JSON or Form):**
    - `grant_type`: OAuth2 grant type - "authorization_code" or "refresh_token" (required)
    - `code`: Authorization code from OAuth2 callback (required for authorization_code grant)
    - `redirect_uri`: Redirect URI used in authorization (required for authorization_code grant)
    - `client_type`: Client type - "papr_plugin" or "browser_extension" (optional, default: papr_plugin)
    - `refresh_token`: Refresh token for token refresh (required for refresh_token grant)
    
    **Response:**
    - `access_token`: OAuth2 access token for API authentication
    - `token_type`: Token type (Bearer)
    - `expires_in`: Token expiration time in seconds
    - `refresh_token`: Refresh token for getting new access tokens
    - `scope`: OAuth2 scopes granted
    - `user_id`: User ID from Auth0
    
    **Example Request:**
    ```json
    {
        "grant_type": "authorization_code",
        "code": "abc123...",
        "redirect_uri": "https://chat.openai.com",
        "client_type": "papr_plugin"
    }
    ```
    """,
    openapi_extra={
        "operationId": "oauth2_token",
        "tags": ["Authentication"]
    }
)
async def token(request: Request):
    try:
        content_type = request.headers.get('Content-Type', '')
        logger.info(f"[TOKEN] Content-Type: {content_type}")

        if 'application/json' in content_type:
            data = await request.json()
        elif 'application/x-www-form-urlencoded' in content_type:
            form_data = await request.form()
            data = dict(form_data)
        else:
            logger.error("[TOKEN] Unsupported Media Type")
            raise HTTPException(status_code=415, detail="Unsupported Media Type")

        logger.info(f"[TOKEN] Received data: {data}")

        grant_type = data.get('grant_type')
        client_type = data.get('client_type', 'papr_plugin')
        auth_code = data.get('code')
        redirect_uri = data.get('redirect_uri')
        logger.info(f"[TOKEN] grant_type={grant_type}, code={auth_code}, redirect_uri={redirect_uri}, client_type={client_type}")

        if grant_type == 'authorization_code':
            if not auth_code:
                raise HTTPException(status_code=400, detail="No authorization code provided")
                
            if not redirect_uri:
                raise HTTPException(status_code=400, detail="Missing redirect URI")

            # First check if user needs onboarding
            auth0 = get_oauth_client(oauth, client_type)
            try:
                # Get initial tokens to check user status
                initial_tokens = await get_token.authorization_code(
                    client_type, 
                    auth_code, 
                    redirect_uri
                )
                
                if 'access_token' in initial_tokens:
                    user_info = await get_token.get_user_info(initial_tokens['access_token'])
                    
                    # Check if user needs onboarding
                    completed_profile_signup = user_info.get('https://papr.scope.com/completedProfileSignup', False)
                    email_verified = user_info.get('email_verified', False)
                    
                    if not completed_profile_signup or not email_verified:
                        logger.info(f"User needs onboarding - Profile complete: {completed_profile_signup}, Email verified: {email_verified}")
                        # Generate return token for web app
                        return_token = secrets.token_urlsafe(32)
                        
                        # Store tokens and info for later use
                        request.session[f'return_info_{return_token}'] = {
                            'tokens': initial_tokens,
                            'user_info': user_info,
                            'redirect_uri': redirect_uri,
                            'auth_code': auth_code,  # Store the original auth code
                            'expires': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
                        }
                        
                        # Generate a new state for the return journey
                        new_state = secrets.token_urlsafe(16)
                        request.session[f'onboarding_state_{new_state}'] = {
                            'return_token': return_token,
                            'original_code': auth_code,
                            'expires': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
                        }
                    
                    # Log amplitude and return tokens (for both onboarded and non-onboarded users)
                    await log_amplitude_event(
                        event_type="login",
                        user_info=user_info,
                        client_type=client_type,
                        amplitude_client=amplitude_client,
                        logger=logger
                    )
                    return initial_tokens
                    
            except Exception as e:
                logger.error(f"Error during token exchange: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))

        elif grant_type == 'refresh_token':
            refresh_token = data.get('refresh_token')
            try:
                new_tokens = await get_token.refresh_token(client_type, refresh_token)
                if 'access_token' in new_tokens:
                    user_info = await get_token.get_user_info(new_tokens['access_token'])
                    await log_amplitude_event(
                        event_type="refresh",
                        user_info=user_info,
                        client_type=client_type,
                        amplitude_client=amplitude_client,
                        logger=logger
                    )
                    return new_tokens
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        elif grant_type == 'client_credentials':
            client_id = data.get('client_id')
            client_secret = data.get('client_secret')
            
            if not client_id or not client_secret:
                raise HTTPException(
                    status_code=400, 
                    detail="Both client_id and client_secret are required for client_credentials grant type"
                )
            
            try:
                tokens = await get_token.client_credentials(client_id, client_secret)
                if 'access_token' in tokens:
                    await log_amplitude_event(
                        event_type="api_client_auth",
                        user_info={"client_id": client_id},
                        client_type="api_client",
                        amplitude_client=amplitude_client,
                        logger=logger
                    )
                    return tokens
            except Exception as e:
                raise HTTPException(status_code=401, detail="Invalid client credentials")

        raise HTTPException(status_code=400, detail="Invalid grant type")

    except Exception as e:
        logger.error(f"Error during token lookup: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/me", 
    response_model=UserInfoResponse,
    responses={
        200: {"model": UserInfoResponse, "description": "User information retrieved successfully"},
        401: {"model": ErrorResponse, "description": "Unauthorized - invalid authentication"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    description="""Get current user information. Validates authentication and returns user details.
    
    **Authentication Required:**
    One of the following authentication methods must be used:
    - Bearer token in `Authorization` header: `Authorization: Bearer <access_token>`
    - Session token in `Authorization` header: `Authorization: Session <session_token>`
    - API Key in `Authorization` header: `Authorization: APIKey <api_key>`
    
    **Headers:**
    - `Authorization`: Authentication token (required)
    - `X-Client-Type`: Client type for logging (optional, default: papr_plugin)
    
    **Response:**
    - `user_id`: Internal user ID
    - `sessionToken`: Session token for API access (if available)
    - `imageUrl`: User profile image URL (if available)
    - `displayName`: User display name (if available)
    - `email`: User email address (if available)
    - `message`: Authentication status message
    
    **Example:**
    ```
    GET /me
    Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...
    X-Client-Type: papr_plugin
    ```
    """,
    openapi_extra={
        "operationId": "get_user_info",
        "tags": ["Authentication"]
    }
)
async def me(request: Request):
    auth_header = request.headers.get('Authorization')
    client_type = request.headers.get('X-Client-Type', 'papr_plugin')

    if not auth_header or ('Bearer ' not in auth_header and 'Session ' not in auth_header and 'APIKey ' not in auth_header):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    user_id = None
    session_token = None
    image_url = None
    display_name = None
    email = None
    user_info = None

    if 'Bearer ' in auth_header:
        token = auth_header.split('Bearer ')[1]
        user_info = await User.verify_access_token(token, client_type)
        if not user_info:
            raise HTTPException(status_code=401, detail="Invalid access token")

        user_id = user_info['https://papr.scope.com/objectId']
        session_token = user_info['https://papr.scope.com/sessionToken']
        image_url = user_info['https://papr.scope.com/profileImage']
        display_name = user_info['https://papr.scope.com/displayName']
        email = user_info['email']

    elif 'Session ' in auth_header:
        session_token = auth_header.split('Session ')[1]
        user = await User.verify_session_token(session_token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid session token")

        user_id = user.get_id()
        user_info = user
        # Extract other user info from Parse Server user object
        image_url = getattr(user, 'profile_image', None)
        display_name = getattr(user, 'display_name', None)
        email = getattr(user, 'email', None)

    elif 'APIKey ' in auth_header:
        api_key = auth_header.split('APIKey ')[1]
        user_info = await User.verify_api_key(api_key)
        if not user_info:
            raise HTTPException(status_code=401, detail="Invalid API key")
        user_id = user_info.get_id()
        image_url = getattr(user, 'profile_image', None)
        display_name = getattr(user, 'display_name', None)
        email = getattr(user, 'email', None)

    await log_amplitude_event(
        event_type="me",
        user_info=user_info,
        client_type=client_type,
        amplitude_client=amplitude_client,
        logger=logger
    )

    return {
        "user_id": user_id,
        "sessionToken": session_token,
        "imageUrl": image_url,
        "displayName": display_name,
        "email": email,
        "message": "You are authenticated!"
    }

@app.get("/logout", 
    response_model=LogoutResponse,
    responses={
        200: {"model": LogoutResponse, "description": "Logout initiated successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    description="""OAuth2 logout endpoint. Logs out the user from Auth0 and redirects to specified URL.
    
    **Query Parameters:**
    - `returnTo`: URL to redirect to after logout (optional, default: extension logout page)
    - `client_type`: Client type for determining Auth0 client ID (optional, default: papr_plugin)
    
    **Flow:**
    1. Client redirects user to this endpoint
    2. This endpoint redirects to Auth0 logout URL
    3. Auth0 logs out the user and redirects to the specified return URL
    
    **Example:**
    ```
    GET /logout?returnTo=https://chat.openai.com
    ```
    
    **Note:** This endpoint initiates the logout process. The actual logout completion happens on Auth0's side.
    """,
    openapi_extra={
        "operationId": "oauth2_logout",
        "tags": ["Authentication"]
    }
)
async def logout(request: Request):
    client_type = request.headers.get('X-Client-Type', 'papr_plugin')
    
    client_id = env.get("AUTH0_CLIENT_ID_PAPR") if client_type == 'papr_plugin' else env.get("AUTH0_CLIENT_ID_BROWSER")
    
    return_to_url = request.query_params.get("returnTo", "extension://ihidpcjponiilpjdpchbgkenhfmdaoil/logout.html")
    
    logout_url = (
        f"https://{env.get('AUTH0_DOMAIN')}/v2/logout?"
        + urlencode(
            {
                "returnTo": return_to_url,
                "client_id": client_id,
            },
            quote_via=quote_plus,
        )
    )
    
    return RedirectResponse(url=logout_url)



@app.get("/neo4j-health", include_in_schema=not HIDE_INTERNAL_ENDPOINTS)
async def neo4j_health(memory_graph: MemoryGraph = Depends(get_memory_graph)):
    try:
        # Check Neo4j connection health - returns True if healthy, False if degraded
        neo4j_healthy = await memory_graph.check_neo4j_health()
        # We should continue even if unhealthy, as we want to return the degraded status
        # and additional diagnostic information to help debug connection issues
        
        # Add test query
        test_query_result = None
        try:
            async with memory_graph.async_neo_conn.get_session() as session:
                result = await session.run("RETURN datetime() as now")
                record = await result.single()
                test_query_result = str(record["now"]) if record else None
        except Exception as e:
            test_query_result = f"Query failed: {str(e)}"
            
        ssl_config = {
            "SSL_CERT_FILE": os.environ.get("SSL_CERT_FILE"),
            "REQUESTS_CA_BUNDLE": os.environ.get("REQUESTS_CA_BUNDLE"),
            "NODE_TLS_REJECT_UNAUTHORIZED": os.environ.get("NODE_TLS_REJECT_UNAUTHORIZED"),
            "using_certifi": bool(certifi.where()),
            "certifi_path": certifi.where(),
            "connection_mode": "Secure" if NEO4J_URL.startswith(('neo4j+s://', 'bolt+s://', 'neo4j+ssc://', 'bolt+ssc://')) else "Standard"       
        }
        
        # Fix URL parsing to correctly extract host
        url_parts = NEO4J_URL.split("@")
        host = url_parts[-1].split(":")[0] if len(url_parts) > 1 else NEO4J_URL.split("://")[1].split(":")[0]

        status_code = 200 if neo4j_healthy else 207  # 207 for degraded service
        
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "healthy" if neo4j_healthy else "degraded",
                "ssl_configuration": ssl_config,
                "neo4j_url": NEO4J_URL.split("@")[0] + "@****",
                "fallback_mode": memory_graph.async_neo_conn.fallback_mode if memory_graph.async_neo_conn else None,
                "connection_details": {
                    "protocol": NEO4J_URL.split("://")[0],
                    "host": host
                },
                "test_query": test_query_result
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "ssl_configuration": ssl_config
            }
        )

@app.get("/redis-health", include_in_schema=not HIDE_INTERNAL_ENDPOINTS)
async def redis_health():
    """Check Redis connection health - Redis is not used in this deployment"""
    return {
        "status": "disabled",
        "message": "Redis is not configured or used in this deployment"
    }





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)