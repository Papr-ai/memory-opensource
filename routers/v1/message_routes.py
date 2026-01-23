"""
Message routes for chat message storage and processing
"""
from fastapi import APIRouter, HTTPException, Request, Depends, Response, BackgroundTasks, Query, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
from typing import Optional
import httpx
import time

from models.message_models import MessageRequest, MessageResponse, MessageHistoryResponse, ConversationSummaryResponse, SessionSummaryResponse
from services.message_service import store_message_in_parse, get_session_messages, get_unprocessed_messages_for_session, get_or_create_chat_session
from services.message_processing_pipeline import add_message_processing_task
from services.auth_utils import get_user_from_token_optimized
from services.multi_tenant_utils import extract_multi_tenant_context
from services.logger_singleton import LoggerSingleton
from services.utils import log_amplitude_event, get_memory_graph
from memory.memory_graph import MemoryGraph

logger = LoggerSingleton.get_logger(__name__)

# Security schemes
bearer_auth = HTTPBearer(scheme_name="Bearer", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
session_token_header = APIKeyHeader(name="X-Session-Token", auto_error=False)

router = APIRouter(prefix="/messages", tags=["Messages"])


@router.post("",
    response_model=MessageResponse,
    responses={
        200: {"model": MessageResponse, "description": "Message stored and queued for processing"},
        400: {"description": "Bad request"},
        401: {"description": "Unauthorized"},
        500: {"description": "Internal server error"}
    },
    description="""
    Store a chat message and queue it for AI analysis and memory creation.
    
    **Authentication Required**: Bearer token, API key, or session token
    
    **Processing Control**:
    - Set `process_messages: true` (default) to enable full AI analysis and memory creation
    - Set `process_messages: false` to store messages only without processing into memories
    
    **Processing Flow** (when process_messages=true):
    1. Message is immediately stored in PostMessage class
    2. Background processing analyzes the message for memory-worthiness
    3. If worthy, creates a memory with appropriate role-based categorization
    4. Links the message to the created memory
    
    **Role-Based Categories**:
    - **User messages**: preference, task, goal, facts, context
    - **Assistant messages**: skills, learning
    
    **Session Management**:
    - `sessionId` is required to group related messages
    - Use the same `sessionId` for an entire conversation
    - Retrieve conversation history using GET /messages/sessions/{sessionId}
    """
)
async def store_message(
    request: Request,
    message_request: MessageRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Security(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Security(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
) -> MessageResponse:
    """Store a chat message and queue for processing"""
    
    try:
        # Handle auth header patching (same as memory endpoint)
        auth_header = request.headers.get('Authorization')
        if not auth_header and api_key:
            # Patch the request headers to include Authorization for downstream code
            request.headers.__dict__['_list'].append(
                (b'authorization', f'APIKey {api_key}'.encode())
            )
            auth_header = request.headers.get('Authorization')

        if not auth_header and not api_key and not session_token:
            response.status_code = 401
            raise HTTPException(status_code=401, detail="Missing authentication")

        # --- Optimized authentication using cached method ---
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')
        auth_start_time = time.time()
        auth_response = None
        
        async with httpx.AsyncClient() as httpx_client:
            # Use optimized authentication (same pattern as memory endpoint)
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
                auth_response = await get_user_from_token_optimized(auth_header, client_type, memory_graph, httpx_client=httpx_client)
    
    except ValueError as e:
        logger.error(f"Invalid authentication token: {e}")
        response.status_code = 401
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        response.status_code = 401
        raise HTTPException(status_code=401, detail="Authentication failed")
        
    if not auth_response:
        response.status_code = 401
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        
        # Extract API key ID from auth_response if available
        api_key_id = None
        if auth_response.api_key_info and isinstance(auth_response.api_key_info, dict):
            api_key_doc = auth_response.api_key_info.get('api_key_doc')
            if isinstance(api_key_doc, dict):
                api_key_id = api_key_doc.get('_id') or api_key_doc.get('objectId')

        # Extract multi-tenant context
        multi_tenant_context = extract_multi_tenant_context(auth_response)
        
        # Store message in Parse Server
        message_response = await store_message_in_parse(
            message_request=message_request,
            user_id=auth_response.end_user_id,
            workspace_id=auth_response.workspace_id,
            organization_id=multi_tenant_context.get("organization_id"),
            namespace_id=multi_tenant_context.get("namespace_id")
        )
        
        # Queue background processing
        add_message_processing_task(
            background_tasks=background_tasks,
            message_request=message_request,
            message_response=message_response,
            user_id=auth_response.end_user_id,
            session_token=auth_response.session_token or "",
            workspace_id=auth_response.workspace_id,
            organization_id=multi_tenant_context.get("organization_id"),
            namespace_id=multi_tenant_context.get("namespace_id"),
            api_key_id=api_key_id
        )
        
        # Log analytics event
        await log_amplitude_event(
            event_type="message_stored",
            user_info={"user_id": auth_response.end_user_id},
            client_type="papr_plugin",
            amplitude_client=None,  # Deprecated
            logger=logger,
            extra_properties={
                "role": message_request.role.value,
                "session_id": message_request.sessionId,
                "content_length": len(message_request.content),
                "has_metadata": message_request.metadata is not None
            },
            end_user_id=auth_response.end_user_id
        )
        
        logger.info(f"Message stored successfully: {message_response.objectId}")
        return message_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing message: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sessions/{session_id}",
    response_model=MessageHistoryResponse,
    responses={
        200: {"model": MessageHistoryResponse, "description": "Message history retrieved"},
        400: {"description": "Bad request"},
        401: {"description": "Unauthorized"},
        404: {"description": "Session not found"},
        500: {"description": "Internal server error"}
    },
    description="""
    Retrieve message history for a specific conversation session.
    
    **Authentication Required**: Bearer token, API key, or session token
    
    **Pagination**:
    - Use `limit` and `skip` parameters for pagination
    - Messages are returned in chronological order (oldest first)
    - `total_count` indicates total messages in the session
    
    **Summaries** (if available):
    - Returns hierarchical conversation summaries (short/medium/long-term)
    - Includes `context_for_llm` field with pre-compressed context
    - Summaries are automatically generated every 15 messages
    - Use `/sessions/{session_id}/summarize` endpoint to generate on-demand
    
    **Access Control**:
    - Only returns messages for the authenticated user
    - Workspace scoping is applied if available
    """
)
async def get_session_history(
    request: Request,
    session_id: str,
    response: Response,
    limit: int = Query(50, ge=1, le=100, description="Maximum number of messages to return"),
    skip: int = Query(0, ge=0, description="Number of messages to skip for pagination"),
    api_key: Optional[str] = Security(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Security(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
) -> MessageHistoryResponse:
    """Retrieve message history for a session"""

    try:
        # Use the injected singleton memory graph
        # Authenticate user
        if api_key and bearer_token:
            auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", "papr_plugin", memory_graph, api_key=api_key)
        elif api_key and session_token:
            auth_response = await get_user_from_token_optimized(f"Session {session_token}", "papr_plugin", memory_graph, api_key=api_key)
        elif api_key:
            auth_response = await get_user_from_token_optimized(f"APIKey {api_key}", "papr_plugin", memory_graph, api_key=api_key)
        elif bearer_token:
            auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", "papr_plugin", memory_graph)
        elif session_token:
            auth_response = await get_user_from_token_optimized(f"Session {session_token}", "papr_plugin", memory_graph)
        else:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        if not auth_response:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Get session messages
        message_history = await get_session_messages(
            session_id=session_id,
            user_id=auth_response.end_user_id,
            limit=limit,
            skip=skip,
            workspace_id=auth_response.workspace_id
        )
        
        # Log analytics event
        await log_amplitude_event(
            event_type="session_history_retrieved",
            user_info={"user_id": auth_response.end_user_id},
            client_type="papr_plugin",
            amplitude_client=None,  # Deprecated
            logger=logger,
            extra_properties={
                "session_id": session_id,
                "message_count": len(message_history.messages),
                "total_count": message_history.total_count,
                "limit": limit,
                "skip": skip
            },
            end_user_id=auth_response.end_user_id
        )
        
        logger.info(f"Retrieved {len(message_history.messages)} messages for session {session_id}")
        return message_history
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session history: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sessions/{session_id}/status",
    responses={
        200: {"description": "Session processing status"},
        401: {"description": "Unauthorized"},
        404: {"description": "Session not found"},
        500: {"description": "Internal server error"}
    },
    description="""
    Get processing status for messages in a session.
    
    **Authentication Required**: Bearer token, API key, or session token
    
    **Status Information**:
    - Total messages in session
    - Processing status breakdown (queued, analyzing, completed, failed)
    - Any messages with processing errors
    """
)
async def get_session_status(
    request: Request,
    session_id: str,
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    api_key: Optional[str] = Depends(api_key_header),
    session_token: Optional[str] = Depends(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
):
    """Get processing status for a session"""

    try:
        # Use the injected singleton memory graph
        # Authenticate user
        if api_key and bearer_token:
            auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", "papr_plugin", memory_graph, api_key=api_key)
        elif api_key and session_token:
            auth_response = await get_user_from_token_optimized(f"Session {session_token}", "papr_plugin", memory_graph, api_key=api_key)
        elif api_key:
            auth_response = await get_user_from_token_optimized(f"APIKey {api_key}", "papr_plugin", memory_graph, api_key=api_key)
        elif bearer_token:
            auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", "papr_plugin", memory_graph)
        elif session_token:
            auth_response = await get_user_from_token_optimized(f"Session {session_token}", "papr_plugin", memory_graph)
        else:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        if not auth_response:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Get all messages for the session (no limit for status check)
        message_history = await get_session_messages(
            session_id=session_id,
            user_id=auth_response.end_user_id,
            limit=1000,  # High limit to get all messages
            skip=0,
            workspace_id=auth_response.workspace_id
        )
        
        # Analyze processing status
        status_counts = {
            "pending": 0,  # Added: messages queued for batch processing
            "stored_only": 0,  # Added: messages stored without processing
            "queued": 0,
            "analyzing": 0,
            "completed": 0,
            "failed": 0,
            "unknown": 0
        }

        failed_messages = []

        for message in message_history.messages:
            status = message.processing_status
            if status in status_counts:
                status_counts[status] += 1
            else:
                status_counts["unknown"] += 1

            if status == "failed":
                failed_messages.append({
                    "message_id": message.objectId,
                    "content_preview": message.content[:100] + "..." if len(message.content) > 100 else message.content,
                    "role": message.role
                })

        session_status = {
            "session_id": session_id,
            "message_count": message_history.total_count,  # Changed from total_messages
            "processing_summary": status_counts,  # Changed from status_breakdown
            "failed_messages": failed_messages,
            "processing_complete": status_counts["pending"] + status_counts["queued"] + status_counts["analyzing"] == 0
        }
        
        logger.info(f"Session status retrieved for {session_id}: {status_counts}")
        return session_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/sessions/{session_id}/summarize",
    response_model=SessionSummaryResponse,
    responses={
        200: {"model": SessionSummaryResponse, "description": "Session summary generated or retrieved"},
        401: {"description": "Unauthorized"},
        404: {"description": "Session not found"},
        500: {"description": "Internal server error"}
    },
    description="""
    Generate or retrieve conversation summaries for a session.
    
    **Authentication Required**: Bearer token, API key, or session token
    
    **Behavior**:
    - If summaries already exist, returns them immediately
    - If no summaries exist, triggers summarization now
    - Returns hierarchical summaries (short/medium/long-term) and topics
    - Includes AI agent instructions for searching memory by sessionId
    
    **Use Cases**:
    - Get summaries for sessions that haven't hit the 15-message threshold yet
    - Provide compressed context to AI agents
    - Quick overview of long conversations
    
    **AI Agent Note**:
    - Response includes instructions for searching memories by sessionId
    - Use the metadata filter: `sessionId='<session_id>'` to find related memories
    """
)
async def summarize_session(
    request: Request,
    session_id: str,
    background_tasks: BackgroundTasks,
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    api_key: Optional[str] = Depends(api_key_header),
    session_token: Optional[str] = Depends(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
):
    """Generate or retrieve conversation summaries for a session"""
    
    try:
        # Authenticate user
        async with httpx.AsyncClient() as httpx_client:
            client_type = "papr_plugin"
            auth_header = request.headers.get("Authorization", "")
            
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
                auth_response = await get_user_from_token_optimized(auth_header, client_type, memory_graph, httpx_client=httpx_client)
        
        if not auth_response:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Extract multi-tenant context
        multi_tenant_context = extract_multi_tenant_context(auth_response)
        
        # First, check if summaries already exist by getting the chat session
        chat = await get_or_create_chat_session(
            session_id=session_id,
            user_id=auth_response.end_user_id,
            workspace_id=auth_response.workspace_id,
            organization_id=multi_tenant_context.get("organization_id"),
            namespace_id=multi_tenant_context.get("namespace_id")
        )
        
        if not chat:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Check if summaries already exist in Parse Server
        existing_summaries = chat.get("summaries")
        
        if existing_summaries and existing_summaries.get("short_term"):
            # Summaries exist, return them with AI agent instructions
            logger.info(f"Returning existing summaries for session {session_id}")
            
            return SessionSummaryResponse(
                session_id=session_id,
                summaries=ConversationSummaryResponse(
                    short_term=existing_summaries.get("short_term", ""),
                    medium_term=existing_summaries.get("medium_term", ""),
                    long_term=existing_summaries.get("long_term", ""),
                    topics=existing_summaries.get("topics", []),
                    last_updated=existing_summaries.get("last_updated")
                ),
                ai_agent_note=f"To find more details about this conversation, search memories with metadata filter: sessionId='{session_id}'",
                from_cache=True
            )
        
        # No summaries exist, trigger summarization now
        logger.info(f"No existing summaries found, triggering summarization for session {session_id}")
        
        # Get all messages from the session
        from services.message_service import get_session_messages as get_all_messages
        message_history = await get_all_messages(
            session_id=session_id,
            user_id=auth_response.end_user_id,
            workspace_id=auth_response.workspace_id,
            organization_id=multi_tenant_context.get("organization_id"),
            namespace_id=multi_tenant_context.get("namespace_id"),
            limit=1000  # Get all messages for summarization
        )
        
        if message_history.total_count == 0:
            raise HTTPException(status_code=404, detail="No messages found in session")
        
        # Convert messages to format expected by analyze_message_batch_for_memory
        messages_for_analysis = []
        for msg in message_history.messages:
            messages_for_analysis.append({
                "objectId": msg.objectId,
                "message": msg.content,
                "messageRole": msg.role,
                "role": msg.role,
                "createdAt": msg.createdAt.isoformat() if msg.createdAt else None
            })
        
        # Analyze messages and generate summaries
        from services.message_batch_analysis import analyze_message_batch_for_memory
        analysis_results, summaries = await analyze_message_batch_for_memory(
            messages=messages_for_analysis,
            session_context=f"Session {session_id}",
            session_id=session_id,
            user_id=auth_response.end_user_id,
            organization_id=multi_tenant_context.get("organization_id"),
            namespace_id=multi_tenant_context.get("namespace_id")
        )
        
        # Store summaries in Parse Server
        from services.message_service import update_chat_summaries
        await update_chat_summaries(
            session_id=session_id,
            user_id=auth_response.end_user_id,
            summaries={
                "short_term": summaries.short_term,
                "medium_term": summaries.medium_term,
                "long_term": summaries.long_term,
                "topics": summaries.topics
            },
            workspace_id=auth_response.workspace_id
        )
        
        # Also create/update MessageSession node in Neo4j (in background)
        if summaries:
            from services.message_batch_analysis import process_batch_analysis_results
            background_tasks.add_task(
                process_batch_analysis_results,
                analysis_results,
                summaries,
                auth_response.end_user_id,
                auth_response.session_token or "",
                auth_response.workspace_id,
                multi_tenant_context.get("organization_id"),
                multi_tenant_context.get("namespace_id"),
                None,  # api_key_id
                None,  # project_id
                None,  # goal_id
                session_id,
                None   # parent_background_tasks
            )
        
        logger.info(f"Generated new summaries for session {session_id}")
        
        return SessionSummaryResponse(
            session_id=session_id,
            summaries=ConversationSummaryResponse(
                short_term=summaries.short_term,
                medium_term=summaries.medium_term,
                long_term=summaries.long_term,
                topics=summaries.topics,
                last_updated=None  # Just generated
            ),
            ai_agent_note=f"To find more details about this conversation, search memories with metadata filter: sessionId='{session_id}'",
            from_cache=False,
            message_count=len(messages_for_analysis)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating session summary: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/sessions/{session_id}/process",
    responses={
        200: {"description": "Session messages queued for processing"},
        400: {"description": "Bad request"},
        401: {"description": "Unauthorized"},
        404: {"description": "Session not found"},
        500: {"description": "Internal server error"}
    },
    description="""
    Process all stored messages in a session that were previously stored with process_messages=false.
    
    **Authentication Required**: Bearer token, API key, or session token
    
    This endpoint allows you to retroactively process messages that were initially stored 
    without processing. Useful for:
    - Processing messages after deciding you want them as memories
    - Batch processing large conversation sessions
    - Re-processing sessions with updated AI models
    
    **Processing Behavior**:
    - Only processes messages with status 'stored_only' or 'pending'
    - Uses the same smart batch analysis (every 15 messages)
    - Respects existing memory creation pipeline
    """
)
async def process_session_messages(
    session_id: str,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    api_key: Optional[str] = Depends(api_key_header),
    session_token: Optional[str] = Depends(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
):
    """Process all stored messages in a session into memories"""

    try:
        # Use the injected singleton memory graph
        # Authenticate user
        if api_key and bearer_token:
            auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", "papr_plugin", memory_graph, api_key=api_key)
        elif api_key and session_token:
            auth_response = await get_user_from_token_optimized(f"Session {session_token}", "papr_plugin", memory_graph, api_key=api_key)
        elif api_key:
            auth_response = await get_user_from_token_optimized(f"APIKey {api_key}", "papr_plugin", memory_graph, api_key=api_key)
        elif bearer_token:
            auth_response = await get_user_from_token_optimized(f"Bearer {bearer_token.credentials}", "papr_plugin", memory_graph)
        elif session_token:
            auth_response = await get_user_from_token_optimized(f"Session {session_token}", "papr_plugin", memory_graph)
        else:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        if not auth_response:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Get unprocessed messages for this session
        unprocessed_messages = await get_unprocessed_messages_for_session(
            session_id,
            auth_response.end_user_id
        )
        
        if not unprocessed_messages:
            return {
                "message": "No unprocessed messages found for this session",
                "session_id": session_id,
                "messages_queued": 0
            }
        
        # Queue batch processing for unprocessed messages
        from services.message_batch_analysis import analyze_message_batch_for_memory, process_batch_analysis_results
        
        async def process_session_batch():
            try:
                # Analyze messages in batch
                analysis_results = await analyze_message_batch_for_memory(
                    unprocessed_messages,
                    session_context=f"Retroactive processing for session {session_id}"
                )
                
                # Process the analysis results
                batch_stats = await process_batch_analysis_results(
                    analysis_results,
                    auth_response.end_user_id,
                    auth_response.session_token or "",
                    auth_response.workspace_id,
                    multi_tenant_context.get("organization_id"),
                    multi_tenant_context.get("namespace_id"),
                    api_key_id=api_key_id
                )
                
                logger.info(f"Retroactive processing completed for session {session_id}: {batch_stats}")
                
            except Exception as e:
                logger.error(f"Error in retroactive session processing: {e}")
        
        # Add to background tasks
        background_tasks.add_task(process_session_batch)
        
        return {
            "message": f"Queued {len(unprocessed_messages)} messages for processing",
            "session_id": session_id,
            "messages_queued": len(unprocessed_messages)
        }
        
    except Exception as e:
        logger.error(f"Error processing session messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
