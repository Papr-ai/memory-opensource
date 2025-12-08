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
from models.feedback_models import (
    FeedbackRequest, BatchFeedbackRequest, FeedbackResponse, BatchFeedbackResponse,
    FeedbackType, FeedbackSource
)
from models.parse_server import UserFeedbackLog, ParsePointer
from services.auth_utils import get_user_from_token_optimized
from services.user_utils import User
from services.logger_singleton import LoggerSingleton
from services.memory_management import store_user_feedback_log_async
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

router = APIRouter(prefix="/feedback", tags=["Feedback"])

@router.post("",
    response_model=FeedbackResponse,
    responses={
        200: {
            "model": FeedbackResponse,
            "description": "Feedback submitted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "code": 200,
                        "status": "success",
                        "feedback_id": "fb_123456789",
                        "message": "Feedback submitted successfully and will be processed for model improvement",
                        "error": None,
                        "details": None
                    }
                }
            }
        },
        400: {
            "model": FeedbackResponse,
            "description": "Bad request",
            "content": {
                "application/json": {
                    "example": {
                        "code": 400,
                        "status": "error",
                        "feedback_id": None,
                        "message": "Failed to submit feedback",
                        "error": "Invalid search_id or feedback data",
                        "details": {"field": "search_id", "reason": "Search ID not found"}
                    }
                }
            }
        },
        401: {
            "model": FeedbackResponse,
            "description": "Unauthorized",
            "content": {
                "application/json": {
                    "example": {
                        "code": 401,
                        "status": "error",
                        "feedback_id": None,
                        "message": "Failed to submit feedback",
                        "error": "Missing or invalid authentication",
                        "details": None
                    }
                }
            }
        },
        404: {
            "model": FeedbackResponse,
            "description": "Search ID not found",
            "content": {
                "application/json": {
                    "example": {
                        "code": 404,
                        "status": "error",
                        "feedback_id": None,
                        "message": "Failed to submit feedback",
                        "error": "Search ID abc123def456 not found or access denied",
                        "details": None
                    }
                }
            }
        },
        500: {
            "model": FeedbackResponse,
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "code": 500,
                        "status": "error",
                        "feedback_id": None,
                        "message": "Failed to submit feedback",
                        "error": "Internal server error",
                        "details": {"trace_id": "abc123"}
                    }
                }
            }
        }
    },
    description="""Submit feedback on search results to help improve model performance.
    
    This endpoint allows developers to provide feedback on:
    - Overall answer quality (thumbs up/down, ratings)
    - Specific memory relevance and accuracy
    - User engagement signals (copy, save, create document actions)
    - Corrections and improvements
    
    The feedback is used to train and improve:
    - Router model tier predictions
    - Memory retrieval ranking
    - Answer generation quality
    - Agentic graph search performance
    
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
        "operationId": "submit_feedback",
        "x-openai-isConsequential": False
    }
)
async def submit_feedback_v1(
    request: Request,
    feedback_request: FeedbackRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Depends(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
) -> FeedbackResponse:
    """
    Submit feedback on search results to help improve model performance.
    """
    try:
        # Get client type from headers
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')
        logger.info(f"Feedback submission - client_type: {client_type}")
        
        # --- Optimized authentication using cached method ---
        auth_start_time = time.time()
        try:
            async with httpx.AsyncClient() as httpx_client:
                # Use optimized authentication that gets workspace_id, isQwenRoute, user_roles, user_workspace_ids, and resolves user in parallel
                if api_key and bearer_token:
                    # Developer provides API key + Bearer token for end user - use Bearer token for auth but developer's API key for Parse
                    auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", client_type, memory_graph, api_key=api_key, feedback_request=feedback_request, httpx_client=httpx_client)
                elif api_key and session_token:
                    auth_response = await get_user_from_token_optimized(f"Session {session_token}", client_type, memory_graph, api_key=api_key, feedback_request=feedback_request, httpx_client=httpx_client)
                elif api_key:
                    auth_response = await get_user_from_token_optimized(f"APIKey {api_key}", client_type, memory_graph, feedback_request=feedback_request, httpx_client=httpx_client)
                elif bearer_token:
                    auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", client_type, memory_graph, feedback_request=feedback_request, httpx_client=httpx_client)
                elif session_token:
                    auth_response = await get_user_from_token_optimized(f"Session {session_token}", client_type, memory_graph, feedback_request=feedback_request, httpx_client=httpx_client)
                else:
                    auth_header = request.headers.get('Authorization')
                    auth_response = await get_user_from_token_optimized(auth_header, client_type, memory_graph, feedback_request=feedback_request, httpx_client=httpx_client)
        except ValueError as e:
            logger.error(f"Invalid authentication token: {e}")
            response.status_code = 401
            return FeedbackResponse.failure(
                error="Invalid authentication token",
                code=401
            )
        auth_end_time = time.time()
        logger.info(f"Enhanced authentication timing: {(auth_end_time - auth_start_time) * 1000:.2f}ms")
        
        if not auth_response:
            response.status_code = 401
            return FeedbackResponse.failure(
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
        
        # Check interaction limits (0 mini interaction for submit_feedback - user feedback collection)
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
                operation=MemoryOperationType.SUBMIT_FEEDBACK,
                api_key_id=api_key_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            # For zero-cost operations, this returns None (early return)
            if limit_check:
                response_dict, status_code, is_error = limit_check
                if is_error:
                    response.status_code = status_code
                    return FeedbackResponse.failure(
                        error=response_dict.get('error'),
                        code=status_code
                    )
        
        logger.info(f"Feedback submission - user_id: {user_id}, end_user_id: {end_user_id}")
        logger.info(f"Feedback submission - workspace_id: {workspace_id}")
        
        # Validate that search_id exists in QueryLog
        query_log = await validate_search_id(feedback_request.search_id, user_id, sessionToken, api_key)
        if not query_log:
            response.status_code = 404
            return FeedbackResponse.failure(
                error=f"Search ID {feedback_request.search_id} not found or access denied",
                code=404
            )
        
        # Create feedback record
        feedback_id = str(uuid.uuid4())
        
        # Create UserFeedbackLog from FeedbackData
        user_pointer = ParsePointer(
            objectId=user_id,
            className="_User"
        )
        
        workspace_pointer = ParsePointer(
            objectId=workspace_id,
            className="WorkSpace"
        )
        
        query_log_pointer = ParsePointer(
            objectId=feedback_request.search_id,
            className="QueryLog"
        )
        
        # Create UserFeedbackLog from FeedbackData
        user_feedback_log = UserFeedbackLog(
            objectId=feedback_id,
            user=user_pointer,
            workspace=workspace_pointer,
            queryLog=query_log_pointer,
            userMessage=feedback_request.feedbackData.userMessage,
            assistantMessage=feedback_request.feedbackData.assistantMessage,
            feedbackType=feedback_request.feedbackData.feedbackType,
            feedbackValue=feedback_request.feedbackData.feedbackValue,
            feedbackScore=feedback_request.feedbackData.feedbackScore,
            feedbackText=feedback_request.feedbackData.feedbackText,
            feedbackSource=feedback_request.feedbackData.feedbackSource,
            citedMemoryIds=feedback_request.feedbackData.citedMemoryIds,
            citedNodeIds=feedback_request.feedbackData.citedNodeIds,
            feedbackProcessed=feedback_request.feedbackData.feedbackProcessed,
            feedbackImpact=feedback_request.feedbackData.feedbackImpact
        )
        
        # Submit to Parse Server (background task for performance)
        background_tasks.add_task(
            submit_feedback_to_parse_server, 
            user_feedback_log,
            sessionToken,
            api_key
        )

        # Update Memory counters from feedback citations (if provided)
        try:
            from services.memory_management import update_memory_counters_from_feedback_async
            cited_ids = feedback_request.feedbackData.citedMemoryIds or []
            background_tasks.add_task(
                update_memory_counters_from_feedback_async,
                cited_ids,
                feedback_request.feedbackData.feedbackType,
                feedback_request.feedbackData.feedbackScore,
                sessionToken,
                api_key
            )
        except Exception as e:
            logger.error(f"Failed scheduling memory counters update from feedback: {e}")
        
        # Also update QueryLog with engagement signals
        background_tasks.add_task(
            update_query_log_engagement,
            feedback_request.search_id,
            user_feedback_log.feedbackType,
            user_feedback_log.feedbackScore,
            sessionToken,
            api_key
        )
        
        # Log Amplitude event
        background_tasks.add_task(
            _log_amplitude_event_background,
            event_type="submit_feedback",
            user_info=user_info,
            client_type=client_type,
            amplitude_client=amplitude_client,
            logger=logger,
            api_key=api_key,
            user_id=user_id,
            end_user_id=end_user_id,
            extra_properties={
                'feedback_id': feedback_id,
                'feedback_type': user_feedback_log.feedbackType,
                'feedback_score': user_feedback_log.feedbackScore,
                'search_id': feedback_request.search_id,
                'cited_memory_count': len(user_feedback_log.citedMemoryIds),
                'cited_node_count': len(user_feedback_log.citedNodeIds)
            }
        )
        
        response.status_code = 200
        return FeedbackResponse.success(
            feedback_id=feedback_id,
            message="Feedback submitted successfully and will be processed for model improvement"
        )
        
    except HTTPException as http_ex:
        # Convert HTTPException to FeedbackResponse format while preserving status code
        logger.error(f"HTTPException in submit_feedback: {http_ex.status_code} - {http_ex.detail}")
        response.status_code = http_ex.status_code
        return FeedbackResponse.failure(
            error=http_ex.detail,
            code=http_ex.status_code
        )
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        response.status_code = 500
        return FeedbackResponse.failure(
            error=f"Failed to submit feedback: {str(e)}",
            code=500
        )

@router.post("/batch",
    response_model=BatchFeedbackResponse,
    responses={
        200: {
            "model": BatchFeedbackResponse,
            "description": "Batch feedback processed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "code": 200,
                        "status": "success",
                        "feedback_ids": ["fb_123", "fb_456"],
                        "successful_count": 2,
                        "failed_count": 0,
                        "errors": [],
                        "message": "Processed 2 feedback items successfully"
                    }
                }
            }
        },
        207: {
            "model": BatchFeedbackResponse,
            "description": "Partial success - some feedback items failed",
            "content": {
                "application/json": {
                    "example": {
                        "code": 207,
                        "status": "success",
                        "feedback_ids": ["fb_123"],
                        "successful_count": 1,
                        "failed_count": 1,
                        "errors": [{"index": 1, "search_id": "abc123", "error": "Search ID not found"}],
                        "message": "Processed 1 feedback items successfully"
                    }
                }
            }
        },
        400: {
            "model": BatchFeedbackResponse,
            "description": "Bad request",
            "content": {
                "application/json": {
                    "example": {
                        "code": 400,
                        "status": "error",
                        "feedback_ids": [],
                        "successful_count": 0,
                        "failed_count": 2,
                        "errors": [{"index": 0, "search_id": "abc123", "error": "Invalid feedback data"}],
                        "message": "Batch feedback processed"
                    }
                }
            }
        },
        401: {
            "model": BatchFeedbackResponse,
            "description": "Unauthorized",
            "content": {
                "application/json": {
                    "example": {
                        "code": 401,
                        "status": "error",
                        "feedback_ids": [],
                        "successful_count": 0,
                        "failed_count": 0,
                        "errors": [],
                        "message": "Failed to process batch feedback",
                        "error": "Missing or invalid authentication"
                    }
                }
            }
        },
        500: {
            "model": BatchFeedbackResponse,
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "code": 500,
                        "status": "error",
                        "feedback_ids": [],
                        "successful_count": 0,
                        "failed_count": 0,
                        "errors": [],
                        "message": "Failed to process batch feedback",
                        "error": "Internal server error"
                    }
                }
            }
        }
    },
    description="""Submit multiple feedback items in a single request.
    
    Useful for submitting session-end feedback or bulk feedback collection.
    Each feedback item is processed independently, so partial success is possible.
    
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
        "operationId": "submit_batch_feedback",
        "x-openai-isConsequential": False
    }
)
async def submit_batch_feedback_v1(
    request: Request,
    batch_feedback: BatchFeedbackRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Depends(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
) -> BatchFeedbackResponse:
    """
    Submit multiple feedback items in a single request.
    """
    try:
        # Get client type from headers
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')
        logger.info(f"Batch feedback submission - client_type: {client_type}")
        
        # --- Optimized authentication using cached method ---
        auth_start_time = time.time()
        try:
            async with httpx.AsyncClient() as httpx_client:
                # Use optimized authentication that gets workspace_id, isQwenRoute, user_roles, user_workspace_ids, and resolves user in parallel
                if api_key and bearer_token:
                    # Developer provides API key + Bearer token for end user - use Bearer token for auth but developer's API key for Parse
                    auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", client_type, memory_graph, api_key=api_key, feedback_request=batch_feedback, httpx_client=httpx_client)
                elif api_key and session_token:
                    auth_response = await get_user_from_token_optimized(f"Session {session_token}", client_type, memory_graph, api_key=api_key, feedback_request=batch_feedback, httpx_client=httpx_client)
                elif api_key:
                    auth_response = await get_user_from_token_optimized(f"APIKey {api_key}", client_type, memory_graph, feedback_request=batch_feedback, httpx_client=httpx_client)
                elif bearer_token:
                    auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", client_type, memory_graph, feedback_request=batch_feedback, httpx_client=httpx_client)
                elif session_token:
                    auth_response = await get_user_from_token_optimized(f"Session {session_token}", client_type, memory_graph, feedback_request=batch_feedback, httpx_client=httpx_client)
                else:
                    auth_header = request.headers.get('Authorization')
                    auth_response = await get_user_from_token_optimized(auth_header, client_type, memory_graph, feedback_request=batch_feedback, httpx_client=httpx_client)
        except ValueError as e:
            logger.error(f"Invalid authentication token: {e}")
            response.status_code = 401
            return BatchFeedbackResponse(
                code=401,
                status="error",
                feedback_ids=[],
                successful_count=0,
                failed_count=0,
                errors=[],
                message="Failed to process batch feedback",
                error="Invalid authentication token"
            )
        auth_end_time = time.time()
        logger.info(f"Enhanced authentication timing: {(auth_end_time - auth_start_time) * 1000:.2f}ms")
        
        if not auth_response:
            response.status_code = 401
            return BatchFeedbackResponse(
                code=401,
                status="error",
                feedback_ids=[],
                successful_count=0,
                failed_count=0,
                errors=[],
                message="Failed to process batch feedback",
                error="Missing or invalid authentication"
            )
        
        # Extract user information
        user_id = auth_response.developer_id
        end_user_id = auth_response.end_user_id
        sessionToken = auth_response.session_token
        user_info = auth_response.user_info
        api_key = auth_response.api_key
        workspace_id = auth_response.workspace_id
        
        # Check interaction limits (0 mini interaction for submit_batch_feedback - user feedback collection)
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
                operation=MemoryOperationType.SUBMIT_BATCH_FEEDBACK,
                api_key_id=api_key_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            # For zero-cost operations, this returns None (early return)
            if limit_check:
                response_dict, status_code, is_error = limit_check
                if is_error:
                    response.status_code = status_code
                    return BatchFeedbackResponse(
                        code=status_code,
                        status="error",
                        feedback_ids=[],
                        successful_count=0,
                        failed_count=0,
                        errors=[],
                        message="Failed to process batch feedback",
                        error=response_dict.get('error')
                    )
        
        logger.info(f"Batch feedback submission - user_id: {user_id}, end_user_id: {end_user_id}")
        logger.info(f"Batch feedback submission - workspace_id: {workspace_id}")
        
        feedback_ids = []
        successful_count = 0
        failed_count = 0
        errors = []
        
        for i, feedback in enumerate(batch_feedback.feedback_items):
            try:
                # Validate search_id
                query_log = await validate_search_id(feedback.search_id, user_id, sessionToken, api_key)
                if not query_log:
                    failed_count += 1
                    errors.append({
                        'index': i,
                        'search_id': feedback.search_id,
                        'error': 'Search ID not found or access denied'
                    })
                    continue
                
                feedback_id = str(uuid.uuid4())
                feedback_ids.append(feedback_id)
                
                # Create UserFeedbackLog from FeedbackData
                user_pointer = ParsePointer(
                    objectId=user_id,
                    className="_User"
                )
                
                workspace_pointer = ParsePointer(
                    objectId=workspace_id,
                    className="WorkSpace"
                )
                
                query_log_pointer = ParsePointer(
                    objectId=feedback.search_id,
                    className="QueryLog"
                )
                
                # Create UserFeedbackLog from FeedbackData
                user_feedback_log = UserFeedbackLog(
                    objectId=feedback_id,
                    user=user_pointer,
                    workspace=workspace_pointer,
                    queryLog=query_log_pointer,
                    userMessage=feedback.feedbackData.userMessage,
                    assistantMessage=feedback.feedbackData.assistantMessage,
                    feedbackType=feedback.feedbackData.feedbackType,
                    feedbackValue=feedback.feedbackData.feedbackValue,
                    feedbackScore=feedback.feedbackData.feedbackScore,
                    feedbackText=feedback.feedbackData.feedbackText,
                    feedbackSource=feedback.feedbackData.feedbackSource,
                    citedMemoryIds=feedback.feedbackData.citedMemoryIds,
                    citedNodeIds=feedback.feedbackData.citedNodeIds,
                    feedbackProcessed=feedback.feedbackData.feedbackProcessed,
                    feedbackImpact=feedback.feedbackData.feedbackImpact
                )
                
                # Submit to Parse Server (background task)
                background_tasks.add_task(
                    submit_feedback_to_parse_server,
                    user_feedback_log,
                    sessionToken,
                    api_key
                )

                # Update Memory counters from feedback citations (if provided)
                try:
                    from services.memory_management import update_memory_counters_from_feedback_async
                    cited_ids = feedback.feedbackData.citedMemoryIds or []
                    background_tasks.add_task(
                        update_memory_counters_from_feedback_async,
                        cited_ids,
                        feedback.feedbackData.feedbackType,
                        feedback.feedbackData.feedbackScore,
                        sessionToken,
                        api_key
                    )
                except Exception as e:
                    logger.error(f"Failed scheduling memory counters update from feedback (batch idx={i}): {e}")
                
                # Update QueryLog with engagement signals
                background_tasks.add_task(
                    update_query_log_engagement,
                    feedback.search_id,
                    user_feedback_log.feedbackType,
                    user_feedback_log.feedbackScore,
                    sessionToken,
                    api_key
                )
                
                successful_count += 1
                
            except Exception as e:
                failed_count += 1
                errors.append({
                    'index': i,
                    'search_id': feedback.search_id,
                    'error': str(e)
                })
        
        # Log Amplitude event for batch submission
        background_tasks.add_task(
            _log_amplitude_event_background,
            event_type="submit_batch_feedback",
            user_info=user_info,
            client_type=client_type,
            amplitude_client=amplitude_client,
            logger=logger,
            api_key=api_key,
            user_id=user_id,
            end_user_id=end_user_id,
            extra_properties={
                'batch_size': len(batch_feedback.feedback_items),
                'successful_count': successful_count,
                'failed_count': failed_count,
                'feedback_ids': feedback_ids,
                'session_context': batch_feedback.session_context
            }
        )
        
        # Determine response status code
        if errors and successful_count > 0:
            response.status_code = 207  # Partial success
        elif errors and successful_count == 0:
            response.status_code = 400  # All failed
        else:
            response.status_code = 200  # All successful
        
        return BatchFeedbackResponse.success(
            feedback_ids=feedback_ids,
            successful_count=successful_count,
            failed_count=failed_count,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Error processing batch feedback: {e}", exc_info=True)
        response.status_code = 500
        return BatchFeedbackResponse(
            code=500,
            status="error",
            feedback_ids=[],
            successful_count=0,
            failed_count=0,
            errors=[],
            message="Failed to process batch feedback",
            error=str(e)
        )

@router.get("/{feedback_id}",
    response_model=FeedbackResponse,
    responses={
        200: {
            "model": FeedbackResponse,
            "description": "Feedback retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "code": 200,
                        "status": "success",
                        "feedback_id": "fb_123456789",
                        "message": "Feedback retrieved successfully",
                        "error": None,
                        "details": {
                            "feedback_type": "thumbs_up",
                            "feedback_score": 1,
                            "feedback_text": "This was helpful!",
                            "search_id": "search_123",
                            "created_at": "2024-01-17T17:30:45.123456Z"
                        }
                    }
                }
            }
        },
        401: {
            "model": FeedbackResponse,
            "description": "Unauthorized",
            "content": {
                "application/json": {
                    "example": {
                        "code": 401,
                        "status": "error",
                        "feedback_id": None,
                        "message": "Failed to retrieve feedback",
                        "error": "Missing or invalid authentication",
                        "details": None
                    }
                }
            }
        },
        404: {
            "model": FeedbackResponse,
            "description": "Feedback not found",
            "content": {
                "application/json": {
                    "example": {
                        "code": 404,
                        "status": "error",
                        "feedback_id": None,
                        "message": "Failed to retrieve feedback",
                        "error": "Feedback ID fb_123456789 not found or access denied",
                        "details": None
                    }
                }
            }
        },
        500: {
            "model": FeedbackResponse,
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "code": 500,
                        "status": "error",
                        "feedback_id": None,
                        "message": "Failed to retrieve feedback",
                        "error": "Internal server error",
                        "details": {"trace_id": "abc123"}
                    }
                }
            }
        }
    },
    description="""Retrieve feedback by ID.
    
    This endpoint allows developers to fetch feedback details by feedback ID.
    Only the user who created the feedback or users with appropriate permissions can access it.
    
    **Authentication Required**:
    One of the following authentication methods must be used:
    - Bearer token in `Authorization` header
    - API Key in `X-API-Key` header
    - Session token in `X-Session-Token` header
    
    **Required Headers**:
    - X-Client-Type: (e.g., 'papr_plugin', 'browser_extension')
    """,
    openapi_extra={
        "operationId": "get_feedback_by_id",
        "x-openai-isConsequential": False
    }
)
async def get_feedback_by_id_v1(
    feedback_id: str,
    request: Request,
    response: Response,
    api_key: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Depends(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
) -> FeedbackResponse:
    """
    Retrieve feedback by ID.
    """
    try:
        # Get client type from headers
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')
        logger.info(f"Get feedback by ID - client_type: {client_type}")
        
        # --- Optimized authentication using cached method ---
        auth_start_time = time.time()
        try:
            async with httpx.AsyncClient() as httpx_client:
                # Use optimized authentication that gets workspace_id, isQwenRoute, user_roles, user_workspace_ids
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
                    auth_header = request.headers.get('Authorization')
                    auth_response = await get_user_from_token_optimized(auth_header, client_type, memory_graph, httpx_client=httpx_client)
        except ValueError as e:
            logger.error(f"Invalid authentication token: {e}")
            response.status_code = 401
            return FeedbackResponse.failure(
                error="Invalid authentication token",
                code=401
            )
        auth_end_time = time.time()
        logger.info(f"Enhanced authentication timing: {(auth_end_time - auth_start_time) * 1000:.2f}ms")
        
        if not auth_response:
            response.status_code = 401
            return FeedbackResponse.failure(
                error="Missing or invalid authentication",
                code=401
            )
        
        # Extract user information
        user_id = auth_response.developer_id
        end_user_id = auth_response.end_user_id
        sessionToken = auth_response.session_token
        api_key = auth_response.api_key
        
        # Check interaction limits (1 mini interaction for get_feedback_by_id)
        from models.operation_types import MemoryOperationType
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
                operation=MemoryOperationType.GET_FEEDBACK,
                api_key_id=api_key_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            
            if limit_check:
                response_dict, status_code, is_error = limit_check
                if is_error:
                    response.status_code = status_code
                    return FeedbackResponse.failure(
                        error=response_dict.get('error'),
                        code=status_code
                    )
        
        logger.info(f"Get feedback by ID - user_id: {user_id}, end_user_id: {end_user_id}")
        
        # Retrieve feedback from Parse Server
        feedback_data = await get_feedback_by_id(feedback_id, user_id, sessionToken, api_key)
        if not feedback_data:
            response.status_code = 404
            return FeedbackResponse.failure(
                error=f"Feedback ID {feedback_id} not found or access denied",
                code=404
            )
        
        # Format the response details
        details = {
            "feedback_type": feedback_data.get("feedbackType"),
            "feedback_score": feedback_data.get("feedbackScore"),
            "feedback_text": feedback_data.get("feedbackText"),
            "feedback_source": feedback_data.get("feedbackSource"),
            "search_id": feedback_data.get("queryLog", {}).get("objectId"),
            "cited_memory_ids": feedback_data.get("citedMemoryIds", []),
            "cited_node_ids": feedback_data.get("citedNodeIds", []),
            "created_at": feedback_data.get("createdAt"),
            "updated_at": feedback_data.get("updatedAt")
        }
        
        response.status_code = 200
        return FeedbackResponse.success(
            feedback_id=feedback_id,
            message="Feedback retrieved successfully",
            details=details
        )
        
    except HTTPException as http_ex:
        # Convert HTTPException to FeedbackResponse format while preserving status code
        logger.error(f"HTTPException in get_feedback_by_id: {http_ex.status_code} - {http_ex.detail}")
        response.status_code = http_ex.status_code
        return FeedbackResponse.failure(
            error=http_ex.detail,
            code=http_ex.status_code
        )
    except Exception as e:
        logger.error(f"Error retrieving feedback: {e}", exc_info=True)
        response.status_code = 500
        return FeedbackResponse.failure(
            error=f"Failed to retrieve feedback: {str(e)}",
            code=500
        )

# Helper functions
async def validate_search_id(search_id: str, user_id: str, session_token: str, api_key: Optional[str]) -> Optional[dict]:
    """Validate that search_id exists and user has access"""
    try:
        # Import the function from memory_management
        from services.memory_management import get_query_log_by_id_async
        
        query_log = await get_query_log_by_id_async(
            query_log_id=search_id,
            session_token=session_token,
            api_key=api_key
        )
        
        if not query_log:
            logger.warning(f"QueryLog not found for search_id: {search_id}")
            return None
        
        # Check if user has access to this QueryLog
        query_log_user_id = query_log.get('user', {}).get('objectId')
        if query_log_user_id and query_log_user_id != user_id:
            logger.warning(f"User {user_id} does not have access to QueryLog {search_id} (owned by {query_log_user_id})")
            return None
        
        return query_log
        
    except Exception as e:
        logger.error(f"Error validating search_id {search_id}: {e}", exc_info=True)
        return None

async def submit_feedback_to_parse_server(user_feedback_log: UserFeedbackLog, session_token: str, api_key: Optional[str]):
    """Submit feedback to Parse Server UserFeedbackLog"""
    try:
        result = await store_user_feedback_log_async(
            user_feedback_log=user_feedback_log,
            session_token=session_token,
            api_key=api_key
        )
        
        if result:
            logger.info(f"Successfully stored UserFeedbackLog with objectId: {result.get('objectId')}")
        else:
            logger.error("Failed to store UserFeedbackLog")
            
    except Exception as e:
        logger.error(f"Error storing UserFeedbackLog: {e}", exc_info=True)

async def update_query_log_engagement(search_id: str, feedback_type: FeedbackType, feedback_score: Optional[int], session_token: str, api_key: Optional[str]):
    """Update QueryLog with engagement signals"""
    try:
        # Import the function from memory_management
        from services.memory_management import update_query_log_engagement_async
        
        # Map feedback type to engagement signal
        engagement_signal = None
        if feedback_type in [FeedbackType.THUMBS_UP, FeedbackType.THUMBS_DOWN]:
            engagement_signal = "thumbs_feedback"
        elif feedback_type == FeedbackType.RATING:
            engagement_signal = "rating_feedback"
        elif feedback_type in [FeedbackType.COPY_ACTION, FeedbackType.SAVE_ACTION, FeedbackType.CREATE_DOCUMENT]:
            engagement_signal = "user_action"
        elif feedback_type == FeedbackType.CORRECTION:
            engagement_signal = "correction_feedback"
        elif feedback_type == FeedbackType.REPORT:
            engagement_signal = "report_feedback"
        else:
            engagement_signal = "general_feedback"
        
        await update_query_log_engagement_async(
            query_log_id=search_id,
            engagement_signal=engagement_signal,
            feedback_score=feedback_score,
            session_token=session_token,
            api_key=api_key
        )
        
        logger.info(f"Updated QueryLog {search_id} with engagement signal: {engagement_signal}")
        
    except Exception as e:
        logger.error(f"Error updating QueryLog engagement: {e}", exc_info=True)

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

async def get_feedback_by_id(feedback_id: str, user_id: str, session_token: str, api_key: Optional[str]) -> Optional[dict]:
    """Retrieve feedback by ID from Parse Server"""
    try:
        # Import the function from memory_management
        from services.memory_management import get_user_feedback_log_by_id_async
        
        feedback_data = await get_user_feedback_log_by_id_async(
            feedback_id=feedback_id,
            session_token=session_token,
            api_key=api_key
        )
        
        if not feedback_data:
            logger.warning(f"UserFeedbackLog not found for feedback_id: {feedback_id}")
            return None
        
        # Check if user has access to this feedback
        feedback_user_id = feedback_data.get('user', {}).get('objectId')
        if feedback_user_id and feedback_user_id != user_id:
            logger.warning(f"User {user_id} does not have access to feedback {feedback_id} (owned by {feedback_user_id})")
            return None
        
        return feedback_data
        
    except Exception as e:
        logger.error(f"Error retrieving feedback {feedback_id}: {e}", exc_info=True)
        return None 