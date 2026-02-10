from fastapi import Request, HTTPException, APIRouter, BackgroundTasks, Depends
from services.auth_utils import get_user_from_token, get_user_from_token_optimized
from memory.memory_graph import MemoryGraph
from memory.memory_item import (
    TextMemoryItem, CodeSnippetMemoryItem, DocumentMemoryItem,
    WebpageMemoryItem, CodeFileMemoryItem, MeetingMemoryItem,
    PluginMemoryItem, IssueMemoryItem, CustomerMemoryItem
)
from models.parse_server import UpdateMemoryResponse,  AddMemoryResponse, AddMemoryItem, BatchMemoryError, BatchMemoryResponse, ErrorDetail
from models.memory_models import AddMemoryRequest, BatchMemoryRequest, MemoryMetadata, OptimizedAuthResponse
from typing import List, Optional, Dict, Any, Union, Tuple
import logging
import json
import time
from services.user_utils import User
from amplitude import Amplitude, BaseEvent, Identify, EventOptions
from werkzeug.utils import secure_filename
import os
import magic
from services.processPDF import process_pdf_in_background, save_uploaded_file
from services.connector_service import find_user_by_connector_ids
from services.connector_service import transpose_data_to_memory
from os import environ as env
import asyncio
from functools import partial
from dotenv import find_dotenv, load_dotenv
from redis.asyncio import Redis
from services.logging_config import get_logger
from urllib.parse import urlparse
import uuid
from services.memory_service import handle_incoming_memory
from services.memory_management import upload_file_to_parse, get_document_upload_status
from redis.exceptions import TimeoutError as RedisTimeoutError
from models.parse_server import DocumentUploadStatusType, DocumentUploadStatusResponse
from services.document_status import update_processing_status
from services.logger_singleton import LoggerSingleton
from fastapi import HTTPException, Response
import copy
from neo4j import AsyncSession
import httpx
from services.webhook_service import webhook_service
# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

logger = LoggerSingleton.get_logger(__name__)

# Log at module level to verify logger is working
logger.info("Memory routes module loaded")

# Initialize Amplitude client
client = Amplitude(env.get("AMPLITUDE_API_KEY"))
logger.info(f"Amplitude client initialized with API key: {env.get('AMPLITUDE_API_KEY')}")

# Get API key for hotglue
hotglue_api_key = env.get("HOTGLUE_PAPR_API_KEY")
logger.info(f"hotglue_api_key inside memory_routes.py: {hotglue_api_key}")

# Initialize chat_gpt
chat_gpt = None  # This should be properly initialized based on your application's needs

# Add prefix to keys to avoid conflicts with Celery tasks
REDIS_KEY_PREFIX = env.get('CELERY_REDIS_KEY_PREFIX', 'doc-upload')

import asyncio
import uuid
from typing import Dict, Set, Optional
from datetime import datetime, timedelta

# Global task tracking (in production, use Redis or database)
_background_tasks: Dict[str, asyncio.Task] = {}
_task_status: Dict[str, str] = {}  # 'pending', 'running', 'completed', 'failed'

async def _wait_for_background_processing_completion(
    batch_id: str, 
    timeout_seconds: int = 30,
    poll_interval: float = 0.5
) -> bool:
    """
    Wait for background processing to complete with proper monitoring.
    
    Args:
        batch_id: Unique identifier for this batch
        timeout_seconds: Maximum time to wait
        poll_interval: How often to check task status
    
    Returns:
        True if all tasks completed successfully, False if timeout or failure
    """
    start_time = datetime.now()
    timeout = timedelta(seconds=timeout_seconds)
    
    logger.info(f"Starting background task monitoring for batch {batch_id}")
    
    while datetime.now() - start_time < timeout:
        # Check if all tasks for this batch are completed
        batch_tasks = [task_id for task_id in _task_status.keys() if task_id.startswith(f"{batch_id}_")]
        
        if not batch_tasks:
            logger.warning(f"No background tasks found for batch {batch_id}")
            return True
        
        completed_tasks = sum(1 for task_id in batch_tasks if _task_status[task_id] in ['completed', 'failed'])
        
        if completed_tasks == len(batch_tasks):
            # All tasks are done, check if any failed
            failed_tasks = [task_id for task_id in batch_tasks if _task_status[task_id] == 'failed']
            if failed_tasks:
                logger.error(f"Some background tasks failed for batch {batch_id}: {failed_tasks}")
                return False
            else:
                logger.info(f"All background tasks completed successfully for batch {batch_id}")
                return True
        
        # Log progress
        logger.debug(f"Batch {batch_id}: {completed_tasks}/{len(batch_tasks)} tasks completed")
        await asyncio.sleep(poll_interval)
    
    logger.warning(f"Timeout waiting for background tasks to complete for batch {batch_id}")
    return False

async def _add_monitored_background_task(
    background_tasks: BackgroundTasks,
    task_func,
    batch_id: str,
    task_name: str,
    *args,
    **kwargs
) -> str:
    """
    Add a background task with monitoring capabilities.
    
    Returns:
        Task ID for tracking
    """
    task_id = f"{batch_id}_{task_name}_{uuid.uuid4().hex[:8]}"
    
    async def monitored_task():
        try:
            _task_status[task_id] = 'running'
            logger.info(f"Starting monitored background task: {task_id}")
            
            # Execute the actual task
            result = await task_func(*args, **kwargs)
            
            _task_status[task_id] = 'completed'
            logger.info(f"Completed background task: {task_id}")
            return result
            
        except Exception as e:
            _task_status[task_id] = 'failed'
            logger.error(f"Background task {task_id} failed: {e}")
            raise
    
    # Add the monitored task to background tasks
    background_tasks.add_task(monitored_task)
    _task_status[task_id] = 'pending'
    
    logger.info(f"Added monitored background task: {task_id}")
    return task_id

async def common_add_memory_handler(
    request: Request,
    memory_graph: MemoryGraph, 
    background_tasks: BackgroundTasks,
    neo_session: AsyncSession,
    auth_response: OptimizedAuthResponse,
    memory_request: Optional[AddMemoryRequest] = None, 
    tenant_subtenant: str = None, 
    connector: str = None, 
    stream: str = None, 
    skip_background_processing: bool = False,
    upload_id: Optional[str] = None,
    post_objectId: Optional[str] = None,
    legacy_route: bool = True,
    api_key_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None
) -> AddMemoryResponse:
    """
    Common handler for adding memory items.
    Returns AddMemoryResponse (success or failure).
    """
    try:
        # Defensive check to ensure auth_response is not None
        if not auth_response:
            logger.error("auth_response is None in common_add_memory_handler")
            return AddMemoryResponse.failure(
                error="Authentication system error",
                code=500
            )
        
        client_type = request.headers.get('X-Client-Type', 'papr_plugin')
        logger.info(f"client_type: {client_type}")
        logger.info(f"tenant_subtenant: {tenant_subtenant}")
        logger.info(f"connector: {connector}")
        logger.info(f"stream: {stream}")

        request_api_key = request.headers.get('x-papr-api-key')
        logger.info(f"request_api_key: {request_api_key}")

        content_type = request.headers.get('Content-Type', '')
        logger.info(f"Received content_type from client: {content_type}")

        # Use memory_request for the main data
        metadata = memory_request.metadata.model_dump() if memory_request else {}
        external_user_id = metadata.get("user_id")
        logger.info(f"AddMemoryRequest external_user_id: {external_user_id}")

        # Extract values from auth_response
        user_id = auth_response.developer_id
        developer_user_id = auth_response.developer_id
        end_user_id = auth_response.end_user_id
        sessionToken = auth_response.session_token
        user_info = auth_response.user_info
        api_key = auth_response.api_key
        workspace_id = auth_response.workspace_id
        is_qwen_route = auth_response.is_qwen_route
        user_roles = auth_response.user_roles
        user_workspace_ids = auth_response.user_workspace_ids
        updated_metadata = auth_response.updated_metadata
        
        # Extract API key ID from auth_response if available
        api_key_id = None
        if auth_response.api_key_info and isinstance(auth_response.api_key_info, dict):
            api_key_doc = auth_response.api_key_info.get('api_key_doc')
            if api_key_doc:
                api_key_id = api_key_doc.get('_id') or api_key_doc.get('objectId')
        logger.info(f"Extracted api_key_id: {api_key_id}")
        
        # Use updated_metadata from optimized authentication if available
        if updated_metadata and memory_request:
            memory_request.metadata = updated_metadata
            logger.info(f"Updated memory_request.metadata with optimized auth data: {memory_request.metadata}")
        elif memory_request and memory_request.metadata and workspace_id:
            # Fallback: Update metadata with workspace_id if we got it from authentication
            memory_request.metadata.workspace_id = workspace_id
            logger.info(f"Updated memory_request.metadata with workspace_id: {memory_request.metadata}")
        
        logger.info(f"Using auth_response - end_user_id: {end_user_id}, developer_id: {user_id}")
        logger.info(f"is_qwen_route: {is_qwen_route}")
        logger.info(f"workspace_id: {workspace_id}")

        # Ensure single-add requests carry org/namespace scoping on metadata
        try:
            from services.multi_tenant_utils import extract_multi_tenant_context, apply_multi_tenant_scoping_to_memory_request
            if memory_request is not None:
                auth_ctx = extract_multi_tenant_context(auth_response)
                apply_multi_tenant_scoping_to_memory_request(memory_request, auth_ctx)
                logger.info("Applied multi-tenant scoping to single-add memory_request (organization_id/namespace_id)")
        except Exception as _e:
            # Non-fatal: proceed even if scoping helper unavailable
            pass

        if request_api_key:
            if request_api_key != hotglue_api_key:
                return AddMemoryResponse.failure(error="Unauthorized", code=401)

            if tenant_subtenant:
                parts = tenant_subtenant.split('_')
                if len(parts) > 1:
                    subtenant_id = parts[1]
                else:
                    subtenant_id = None
                tenant_id = parts[0]
            else:
                tenant_id = request.query_params.get('tenant')
                subtenant_id = request.query_params.get('subtenant')

            if not tenant_id:
                return AddMemoryResponse.failure(error="Missing tenant ID", code=400)
            if not subtenant_id:
                return AddMemoryResponse.failure(error="Missing subtenant ID", code=400)
            data_request = memory_request.as_handler_dict() if memory_request else {}
            user_id = subtenant_id
            api_key = None
        
            if content_type == 'application/json': 
                if not api_key:
                    sessionToken = await User.lookup_user_token(user_id)
                    if not sessionToken:
                        return AddMemoryResponse.failure(error="Invalid session token", code=401)
                    logger.info(f"sessionToken: {sessionToken}")
                    user_instance = await User.verify_session_token(sessionToken)
                    logger.info(f"User attributes: {dir(user_instance)}")
                    logger.info(f"User as string: {str(user_instance)}")
                    logger.info(f"called verify session_token_with_parse_server with user: {user_instance}")
                    if not user_instance:
                        return AddMemoryResponse.failure(error="Invalid session token", code=401)
                else:
                    user_instance = await User.verify_api_key(api_key)                    
                    if not user_instance:   
                        return AddMemoryResponse.failure(error="Invalid API key", code=401)

                logger.info(f"User ID from 'id' attribute: {user_instance.id}")
                user_id = user_instance.get_id()
                logger.info(f"User ID from 'get_id' method: {user_id}")

                url_path = f"/add_memory/{tenant_subtenant}/{connector}/{stream}"

                if data_request.get('type') == 'message' and data_request.get('subtype') == 'message_changed':
                    memory_item_dict = await transpose_data_to_memory(data_request, url_path, sessionToken, user_id, tenant_id, update=True, api_key=api_key)
                    message = data_request.get('message')
                    previous_message = data_request.get('previous_message')
                    memory_id = await memory_graph.lookup_memory_by_client_msg_id(previous_message.get('client_msg_id'), neo_session)
                    if memory_id:
                        logger.info(f"content update: {memory_item_dict.get('content')}")
                        metadata = memory_item_dict.get('metadata')
                        if not metadata:
                            metadata = {}
                        if isinstance(metadata, str):
                            try:
                                metadata = json.loads(metadata)
                            except json.JSONDecodeError:
                                return AddMemoryResponse.failure(error="Invalid metadata format", code=400)
                        metadata['user_id'] = str(user_id)                    
                        logger.info(f"metadata with user_id: {metadata}")
                        updated_memory_item: UpdateMemoryResponse = await memory_graph.update_memory_item(session_token=sessionToken, memory_id=memory_id, type=memory_item_dict.get('type'), content=memory_item_dict.get('content'), metadata=metadata, background_tasks=background_tasks, neo_session=neo_session, api_key=api_key, legacy_route=legacy_route)
                        if updated_memory_item:
                            memory_items = [
                                AddMemoryItem(
                                    memoryId=item.memoryId if hasattr(item, 'memoryId') else item['memoryId'],
                                    createdAt=item.createdAt if hasattr(item, 'createdAt') else item['createdAt'],
                                    objectId=item.objectId if hasattr(item, 'objectId') else item['objectId'],
                                    memoryChunkIds=item.memoryChunkIds if hasattr(item, 'memoryChunkIds') else []
                                ) for item in updated_memory_item
                            ]
                            return AddMemoryResponse.success(data=memory_items)
                        else:
                            return AddMemoryResponse.failure(error="Failed to update memory item", code=404)
                    else:
                        return AddMemoryResponse.failure(error="Memory item not found won't update", code=404)
                elif data_request.get('type') == 'message' and data_request.get('subtype') == 'message_deleted':
                    previous_message = data_request.get('previous_message')
                    memory_id = await memory_graph.lookup_memory_by_client_msg_id(previous_message.get('client_msg_id'), neo_session)
                    if memory_id:
                        deleted_memory_item = await memory_graph.delete_memory_item(memory_id, sessionToken, neo_session, False, api_key=api_key, legacy_route=legacy_route)
                        if deleted_memory_item:
                            return AddMemoryResponse.success(data=[AddMemoryItem(
                                memoryId=memory_id,
                                createdAt=deleted_memory_item.createdAt if hasattr(deleted_memory_item, 'createdAt') else deleted_memory_item['createdAt'],
                                objectId=memory_id,
                                memoryChunkIds=deleted_memory_item.memoryChunkIds if hasattr(deleted_memory_item, 'memoryChunkIds') else []
                            )])
                        else:
                            return AddMemoryResponse.failure(error="Memory item was not deleted", code=404)
                    else:
                        return AddMemoryResponse.failure(error="Memory item not found won't delete", code=404)
                else:
                    memory_item_dict = await transpose_data_to_memory(data_request, url_path, sessionToken, user_id, tenant_id, False, api_key=api_key)
                    logger.info(f"memory_item_dict inside common_add_memory_handler: {memory_item_dict}")
                    logger.info(f"Type of memory_item_dict: {type(memory_item_dict)}")
                    logger.info(f"🔥🔥🔥 SKIP_BG_PROCESSING: skip_background_processing={skip_background_processing} in common_add_memory_handler")
                    response = await handle_incoming_memory(memory_item_dict, user_id, developer_user_id, sessionToken, neo_session, None, client_type, memory_graph, background_tasks, skip_background_processing, api_key=request_api_key, legacy_route=legacy_route, workspace_id=workspace_id, api_key_id=api_key_id)
                    logger.info(f"common_add_memory_handler - Response from handle_incoming_memory: {response}")
                    logger.info(f"common_add_memory_handler - Response data: {response.data}")
                    logger.info(f"common_add_memory_handler - First item memoryChunkIds: {response.data[0].memoryChunkIds if response.data else 'No data'}")
                    return response
            else:
                return AddMemoryResponse.failure(error="Unsupported Media Type", code=415)

        else: 

            
            external_user_id = external_user_id if external_user_id else user_id
            logger.info(f"external_user_id: {external_user_id}")

            try:
                if 'application/json' in content_type:
                    # Use memory_request for the main data
                    if isinstance(memory_request, list):
                        response = await handle_multiple_memories(
                            memory_request, 
                            external_user_id, 
                            sessionToken, 
                            neo_session, 
                            user_info, 
                            client_type, 
                            memory_graph, 
                            background_tasks, 
                            skip_background_processing, 
                            api_key, 
                            legacy_route=not is_qwen_route,
                            workspace_id=workspace_id,
                            api_key_id=api_key_id,
                            organization_id=organization_id,
                            namespace_id=namespace_id
                        )
                        return response
                    else:
                        try:
                            logger.info(f"🔥🔥🔥 SKIP_BG_PROCESSING: skip_background_processing={skip_background_processing} in common_add_memory_handler (JSON path)")
                            response = await handle_incoming_memory(
                            memory_request, end_user_id, developer_user_id, sessionToken, neo_session, user_info, client_type, memory_graph, background_tasks, skip_background_processing, api_key=api_key, legacy_route=legacy_route, workspace_id=workspace_id, api_key_id=api_key_id
                        )
                            return response
                        except HTTPException as http_ex:
                            logger.error(f"HTTPException in common_add_memory_handler: {http_ex.status_code} - {http_ex.detail}", exc_info=True)
                            return AddMemoryResponse.failure(error=http_ex.detail, code=http_ex.status_code)
                        except Exception as e:
                            logger.error(f"Error processing memory item in common_add_memory_handler: {str(e)}", exc_info=True)
                            return AddMemoryResponse.failure(error=str(e), code=500)

                elif 'multipart/form-data' in content_type:
                    document_metadata = memory_request.metadata if memory_request else None
                    form = await request.form()
                    files = {k: v for k, v in form.items() if hasattr(v, 'file')}
                    if not files:
                        return AddMemoryResponse.failure(error="No file provided", code=400)

                    memory_items = []
                    total_files = len(files)
                    
                    for idx, (_, file_obj) in enumerate(files.items(), 1):
                        filename = secure_filename(file_obj.filename)
                        logger.info(f"Received file: {filename}")
                        try:
                            upload_id = str(uuid.uuid4())
                            logger.info(f"Generated upload_id: {upload_id}")
                            content = await file_obj.read()
                            mime_type = magic.from_buffer(content, mime=True)
                            if mime_type not in ['application/pdf', 'text/html', 'text/plain']:
                                return AddMemoryResponse.failure(error=f"Unsupported file type: {mime_type}", code=415)
                            file_info = await upload_file_to_parse(
                                file_content=content,
                                filename=filename,
                                content_type=mime_type,
                                session_token=sessionToken,
                                api_key=api_key
                            )
                            if not file_info:
                                return AddMemoryResponse.failure(error="Failed to upload file to Parse Server", code=500)
                            initial_metadata = {
                                'file_url': file_info.file_url,
                                'source_url': file_info.source_url,
                                'name': file_info.name,
                                'mime_type': file_info.mime_type,
                                'upload_id': upload_id
                            }
                            try:
                                user_workspace_ids = await User.get_workspaces_for_user_async(user_id)
                                logger.info(f"Retrieved workspace IDs for user {user_id}: {user_workspace_ids}")
                            except Exception as e:
                                logger.error(f"Error getting workspace IDs: {e}")
                                user_workspace_ids = None
                            # Merge document_metadata fields if provided
                            merged_metadata = dict(initial_metadata)
                            if document_metadata:
                                merged_metadata.update(document_metadata.model_dump(exclude_none=True))
                            initial_data = AddMemoryRequest(
                                content=f"Processing {filename}...",
                                type="DocumentMemoryItem",
                                metadata=MemoryMetadata(**merged_metadata),
                                context=[]
                            )
                            initial_response = await handle_incoming_memory(
                                initial_data, external_user_id, developer_user_id, sessionToken, neo_session, user_info, client_type, 
                                memory_graph, background_tasks, True, user_workspace_ids, api_key=api_key, legacy_route=legacy_route, workspace_id=workspace_id, api_key_id=api_key_id
                            )
                            if not initial_response or not initial_response.data:
                                return AddMemoryResponse.failure(error="Failed to create initial memory item", code=500)
                            memory_items.extend(initial_response.data)
                            memory_object_id = initial_response.data[0].objectId
                            logger.info(f"memory_object_id: {memory_object_id}")
                            logger.info(f"initial_metadata: {initial_metadata}")
                            background_tasks.add_task(
                                process_pdf_in_background,
                                file_url=initial_metadata['file_url'],
                                filename=filename,
                                context=None,
                                upload_id=initial_metadata["upload_id"],
                                update_status_callback=update_processing_status,
                                user_id=external_user_id,
                                session_token=sessionToken,
                                memory_objectId=memory_object_id,
                                workspace_id=None,
                                background_tasks=background_tasks,
                                extract_mode='blocks',
                                batch_size=10,
                                memory_graph=memory_graph,
                                client_type=client_type,
                                user_workspace_ids=user_workspace_ids,
                                post_objectId=post_objectId,
                                api_key=api_key,
                                legacy_route=legacy_route,
                                organization_id=getattr(auth_response, "organization_id", None),
                                namespace_id=getattr(auth_response, "namespace_id", None),
                                api_key_id=api_key_id
                            )
                        except Exception as e:
                            logger.error(f"Error processing file {filename}: {e}")
                            if upload_id:
                                await update_processing_status(
                                    upload_id=upload_id,
                                    filename=filename,
                                    current_page=idx,
                                    total_pages=total_files,
                                    status="error",
                                    error=str(e),
                                    objectId=memory_object_id if 'memory_object_id' in locals() else None,
                                    post_objectId=post_objectId,
                                    file_url=initial_metadata['file_url']
                                )
                            return AddMemoryResponse.failure(error=str(e), code=500)
                    return AddMemoryResponse.success(data=memory_items)

                elif 'application/x-www-form-urlencoded' in content_type:
                    form = await request.form()
                    data = dict(form)
                    try:
                        response = await handle_incoming_memory(
                            data, user_id, developer_user_id, sessionToken, neo_session, user_info, client_type, memory_graph, background_tasks, skip_background_processing, api_key=api_key, legacy_route=legacy_route, workspace_id=workspace_id, api_key_id=api_key_id
                        )
                        return response
                    except HTTPException as http_ex:
                        logger.error(f"HTTPException in form-urlencoded handler: {http_ex.status_code} - {http_ex.detail}")
                        return AddMemoryResponse.failure(error=http_ex.detail, code=http_ex.status_code)
                    except Exception as e:
                        logger.error(f"Error processing form-urlencoded memory: {str(e)}")
                        return AddMemoryResponse.failure(error=str(e), code=500)
                else:
                    return AddMemoryResponse.failure(error="Unsupported Media Type", code=415)

            except ValueError as ve:
                return AddMemoryResponse.failure(error=str(ve), code=400)
            except Exception as e:
                logger.error(f"Memory processing error: {e}", exc_info=True)
                return AddMemoryResponse.failure(error="Partial failure", code=207, details=str(e))

    except ValueError as ve:
        return AddMemoryResponse.failure(error=str(ve), code=400)
    except Exception as e:
        logger.error(f"Unexpected error in memory handler: {e}", exc_info=True)
        return AddMemoryResponse.failure(error="Partial failure", code=207, details=str(e))

async def handle_multiple_memories(
    memory_requests: List[AddMemoryRequest],
    user_id: str,
    sessionToken: str,
    neo_session: AsyncSession,
    user_info: Optional[Dict[str, Any]],
    client_type: str,
    memory_graph: MemoryGraph,
    background_tasks: BackgroundTasks,
    skip_background_processing: bool = False,
    api_key: Optional[str] = None,
    legacy_route: bool = True,
    workspace_id: Optional[str] = None,
    api_key_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None
) -> AddMemoryResponse:
    """
    Handle multiple memory items in parallel for speed and robustness.
    Returns an AddMemoryResponse.
    """
    memory_items = []
    errors = []
    semaphore = asyncio.BoundedSemaphore(10)  # Limit concurrency

    async def process_one(memory_request, legacy_route):
        async with semaphore:
            try:
                # Create a new session for this task
                # --- Neo4j session management: create one session for each task ---
                await memory_graph.ensure_async_connection()
                async with memory_graph.async_neo_conn.get_session() as task_session:
                    response = await handle_incoming_memory(
                        memory_request, user_id, user_id, sessionToken, task_session, user_info, client_type, memory_graph, background_tasks, skip_background_processing, api_key=api_key, legacy_route=legacy_route, workspace_id=workspace_id, api_key_id=api_key_id
                    )
                if response and response.data:
                    return response.data, None
                else:
                    return [], None
            except HTTPException as he:
                logger.error(f"HTTPException in process_one: {he.detail}", exc_info=True)
                return [], str(he.detail)
            except Exception as e:
                logger.error(f"Error processing memory item inside process_one: {e}", exc_info=True)
                return [], str(e)

    tasks = [process_one(memory_request, legacy_route) for memory_request in memory_requests]
    results = await asyncio.gather(*tasks)

    for data, error in results:
        if data:
            memory_items.extend(data)
        if error:
            errors.append(error)

    if not memory_items:
        return AddMemoryResponse.failure(error="Failed to process any memory items", code=500)

    return AddMemoryResponse.success(data=memory_items)


async def batch_common_add_memory_handler(
    request: Request,
    memory_graph: MemoryGraph,
    background_tasks: BackgroundTasks,
    neo_session: AsyncSession,
    auth_response: OptimizedAuthResponse,
    memory_requests: List[AddMemoryRequest],
    tenant_subtenant: Optional[str] = None,
    connector: Optional[str] = None,
    stream: Optional[str] = None,
    skip_background_processing: bool = False,
    upload_id: Optional[str] = None,
    post_objectId: Optional[str] = None,
    legacy_route: bool = True
) -> List[AddMemoryResponse]:
    """
    TRUE BATCH HANDLER - processes multiple memories in a single database transaction.
    This is the full batch architecture that replaces the for-loop based common_add_memory_batch_handler.
    
    Args:
        request: FastAPI Request object
        memory_graph: MemoryGraph instance
        background_tasks: FastAPI BackgroundTasks
        neo_session: Shared Neo4j session for all memories
        auth_response: Authentication and authorization info
        memory_requests: List of AddMemoryRequest objects to process
        tenant_subtenant: Optional tenant/subtenant ID
        connector: Optional connector ID
        stream: Optional stream ID
        skip_background_processing: If True, process synchronously
        upload_id: Optional upload ID
        post_objectId: Optional post object ID
        legacy_route: Whether this is a legacy route
        
    Returns:
        List of AddMemoryResponse objects (one per memory)
    """
    logger.info(f"📦 batch_common_add_memory_handler: Processing {len(memory_requests)} memories in TRUE BATCH MODE")
    
    try:
        # Extract values from auth_response
        user_id = auth_response.developer_id
        end_user_id = auth_response.end_user_id
        sessionToken = auth_response.session_token
        user_info = auth_response.user_info
        api_key = auth_response.api_key
        workspace_id = auth_response.workspace_id
        user_workspace_ids = auth_response.user_workspace_ids
        
        logger.info(f"Using auth_response for batch - end_user_id: {end_user_id}, developer_id: {user_id}")
        logger.info(f"workspace_id: {workspace_id}")
        
        # Use the batch processing method from memory_service
        from services.memory_service import batch_handle_incoming_memories
        
        responses = await batch_handle_incoming_memories(
            memory_requests=memory_requests,
            end_user_id=end_user_id,
            developer_user_id=user_id,
            sessionToken=sessionToken,
            neo_session=neo_session,
            user_info=user_info,
            client_type="api",
            memory_graph=memory_graph,
            background_tasks=background_tasks,
            skip_background_processing=skip_background_processing,
            user_workspace_ids=user_workspace_ids,
            api_key=api_key,
            legacy_route=legacy_route,
            workspace_id=workspace_id,
            api_key_id=auth_response.api_key_id if hasattr(auth_response, 'api_key_id') else None
        )
        
        logger.info(f"✅ batch_common_add_memory_handler: Completed {len(responses)} memories")
        return responses
        
    except Exception as e:
        logger.error(f"❌ Error in batch_common_add_memory_handler: {e}", exc_info=True)
        # Return error responses for all memories
        return [AddMemoryResponse.failure(error=str(e), code=500) for _ in memory_requests]


async def common_add_memory_batch_handler(
    request: Request,
    memory_graph: MemoryGraph,
    background_tasks: BackgroundTasks,
    memory_request_batch: BatchMemoryRequest,
    auth_response: OptimizedAuthResponse,
    tenant_subtenant: Optional[str] = None,
    connector: Optional[str] = None,
    stream: Optional[str] = None,
    batch_size: int = 10,
    skip_background_processing: bool = False,
    upload_id: Optional[str] = None,
    post_objectId: Optional[str] = None,
    legacy_route: bool = True
) -> BatchMemoryResponse:
    """
    Handle batch memory additions with size and subscription limits.
    Always returns BatchMemoryResponse (never ErrorDetail).
    """
    try:
        logger.info(f"=== common_add_memory_batch_handler debug ===")
        logger.info(f"Webhook URL in memory_request_batch: {getattr(memory_request_batch, 'webhook_url', None)}")
        logger.info(f"Webhook secret in memory_request_batch: {getattr(memory_request_batch, 'webhook_secret', None)}")
        logger.info(f"memory_request_batch type: {type(memory_request_batch)}")
        logger.info(f"memory_request_batch attributes: {dir(memory_request_batch)}")
        
        memories = memory_request_batch.memories
        if not isinstance(memories, list):
            return BatchMemoryResponse.failure(
                errors=[BatchMemoryError(index=-1, error="Batch endpoint expects an array of memory items")],
                code=400,
                error="Batch endpoint expects an array of memory items"
            )

        # === NEW: Feature flag-based batch validation ===
        from services.batch_processor import validate_batch_size, should_use_temporal
        
        batch_size_count = len(memories)
        is_valid, error_msg, max_size = await validate_batch_size(batch_size_count)
        
        if not is_valid:
            return BatchMemoryResponse.failure(
                errors=[BatchMemoryError(
                    index=-1, 
                    error=f"Batch size ({batch_size_count}) exceeds limit ({max_size}). {error_msg}"
                )],
                code=413,
                error=f"Batch size exceeds limit",
                details={
                    "batch_size": batch_size_count,
                    "max_allowed": max_size,
                    "message": error_msg
                }
            )
        
        # Check if we should use Temporal (cloud-only, large batches)
        # NOTE: Temporal check is handled at the route level (add_memory_batch_v1)
        # to avoid duplicate workflow triggers. This handler only runs for:
        # 1. Non-temporal batches (small batches or open-source)
        # 2. Batches called from within existing Temporal workflows (skip_background_processing=True)
        
        # === Original validation for content size ===
        MAX_CONTENT_LENGTH = int(env.get('MAX_CONTENT_LENGTH', 15000))
        MAX_TOTAL_BATCH_CONTENT = int(env.get('MAX_TOTAL_BATCH_CONTENT', 750000))
        MAX_TOTAL_BATCH_STORAGE = int(env.get('MAX_TOTAL_BATCH_STORAGE', 750000))

        # Calculate and validate total sizes
        total_content_size = 0
        total_storage_size = 0
        for memory in memories:
            content = memory.content
            content_size = len(content.encode('utf-8'))
            total_content_size += content_size
            total_storage_size += content_size

            if content_size > MAX_CONTENT_LENGTH:
                return BatchMemoryResponse.failure(
                    errors=[BatchMemoryError(index=-1, error=f"Individual content size ({content_size} bytes) exceeds maximum limit of {MAX_CONTENT_LENGTH} bytes")],
                    code=413,
                    error=f"Individual content size ({content_size} bytes) exceeds maximum limit of {MAX_CONTENT_LENGTH} bytes"
                )

        # Validate total batch sizes
        if total_content_size > MAX_TOTAL_BATCH_CONTENT:
            return BatchMemoryResponse.failure(
                errors=[BatchMemoryError(index=-1, error=f"Total batch content size ({total_content_size} bytes) exceeds maximum limit of {MAX_TOTAL_BATCH_CONTENT} bytes")],
                code=413,
                error=f"Total batch content size ({total_content_size} bytes) exceeds maximum limit of {MAX_TOTAL_BATCH_CONTENT} bytes"
            )

        if total_storage_size > MAX_TOTAL_BATCH_STORAGE:
            return BatchMemoryResponse.failure(
                errors=[BatchMemoryError(index=-1, error=f"Total batch storage size ({total_storage_size} bytes) exceeds maximum limit of {MAX_TOTAL_BATCH_STORAGE} bytes")],
                code=413,
                error=f"Total batch storage size ({total_storage_size} bytes) exceeds maximum limit of {MAX_TOTAL_BATCH_STORAGE} bytes"
            )

        # Generate batch ID for tracking at the beginning
        batch_id = str(uuid.uuid4())
        
        responses = []
        errors = []

        # Extract values from auth_response
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
        
        logger.info(f"Using auth_response for batch - end_user_id: {end_user_id}, developer_id: {user_id}")
        logger.info(f"is_qwen_route: {is_qwen_route}")
        logger.info(f"workspace_id: {workspace_id}")

        # --- Parallel batch processing with concurrency limit, using common_add_memory_handler ---
        # Reduce concurrency if there are many memories to prevent Qdrant connection exhaustion
        effective_batch_size = min(batch_size, 5) if len(memories) > 20 else batch_size
        semaphore = asyncio.BoundedSemaphore(effective_batch_size)
        
        if effective_batch_size < batch_size:
            logger.info(f"Reduced batch concurrency from {batch_size} to {effective_batch_size} for large batch of {len(memories)} memories")

        async def process_one(idx, memory_request, legacy_route):
            async with semaphore:
                try:
                    # Validate memory content before processing
                    if not memory_request.content or memory_request.content.strip() == "":
                        return (idx, None, "Memory content cannot be empty")
                    
                    # Create a new Request object for each memory item
                    modified_scope = dict(request.scope)
                    
                    # Copy authentication information if it exists
                    if "auth" in request.scope:
                        modified_scope["auth"] = request.scope["auth"]
                    
                    # Copy headers from the original request
                    headers_list = [(k.lower().encode(), v.encode()) for k, v in request.headers.items()]
                    modified_scope['headers'] = headers_list
                    modified_request = Request(scope=modified_scope)

                    # Create a new session for this task
                    # --- Neo4j session management: create one session for each task ---
                    await memory_graph.ensure_async_connection()
                    async with memory_graph.async_neo_conn.get_session() as task_session:

                        # Call the handler
                        response = await common_add_memory_handler(
                            request=modified_request,
                            memory_graph=memory_graph,
                            background_tasks=background_tasks,
                            neo_session=task_session,
                            auth_response=auth_response,
                            memory_request=memory_request,
                            tenant_subtenant=tenant_subtenant,
                            connector=connector,
                            stream=stream,
                            skip_background_processing=skip_background_processing,
                            upload_id=upload_id,
                            post_objectId=post_objectId,
                            legacy_route=legacy_route
                        )
                        return (idx, response, None)
                except Exception as e:
                    return (idx, None, str(e))

        tasks = [process_one(idx, memory, legacy_route) for idx, memory in enumerate(memories)]
        
        # Add timeout for batch operations to prevent hanging
        per_memory_timeout = int(env.get("BATCH_PROCESSING_TIMEOUT_PER_MEMORY_SECONDS", 45))
        min_timeout = int(env.get("BATCH_PROCESSING_TIMEOUT_MIN_SECONDS", 60))
        max_timeout = int(env.get("BATCH_PROCESSING_TIMEOUT_MAX_SECONDS", 300))
        batch_timeout = max(min_timeout, len(memories) * per_memory_timeout)
        batch_timeout = min(max_timeout, batch_timeout)
        logger.info(f"Starting batch processing with {batch_timeout}s timeout for {len(memories)} memories")
        
        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=batch_timeout)
        except asyncio.TimeoutError:
            logger.error(f"Batch operation timed out after {batch_timeout}s for {len(memories)} memories")
            return BatchMemoryResponse.failure(
                errors=[BatchMemoryError(index=-1, error=f"Batch operation timed out after {batch_timeout}s")],
                code=408,
                error=f"Batch operation timed out after {batch_timeout}s"
            )

        for idx, response, error in results:
            if error:
                errors.append(BatchMemoryError(index=idx, error=error))
            else:
                responses.append(response)

        # Compose the final batch response as before
        batch_response = None
        if errors and responses:
            batch_response = BatchMemoryResponse.partial(
                successful=responses,
                errors=errors,
                total_processed=len(memories),
                total_successful=len(responses),
                total_failed=len(errors),
                total_content_size=total_content_size,
                total_storage_size=total_storage_size
            )
        elif errors and not responses:
            batch_response = BatchMemoryResponse.failure(
                errors=errors,
                code=400,
                error="All batch items failed",
                details={"total_content_size": total_content_size, "total_storage_size": total_storage_size}
            )
        else:
            batch_response = BatchMemoryResponse.success(
                successful=responses,
                total_processed=len(memories),
                total_successful=len(responses),
                total_failed=0,
                total_content_size=total_content_size,
                total_storage_size=total_storage_size
            )
        
        # Send webhook notification if configured
        logger.info(f"Webhook URL check: {memory_request_batch.webhook_url}")
        logger.info(f"Webhook URL type: {type(memory_request_batch.webhook_url)}")
        logger.info(f"Webhook URL is None: {memory_request_batch.webhook_url is None}")
        logger.info(f"Webhook URL is empty string: {memory_request_batch.webhook_url == ''}")
        
        if memory_request_batch.webhook_url:
            logger.info(f"Sending webhook notification to: {memory_request_batch.webhook_url}")
            
            # Check if we should wait for background processing
            webhook_wait_for_background = os.getenv("WEBHOOK_WAIT_FOR_BACKGROUND", "true").lower() == "true"
            
            if skip_background_processing or not webhook_wait_for_background:
                # Send webhook immediately
                logger.info("Sending webhook immediately (skip_background_processing=True or webhook_wait_for_background=False)")
                await _send_batch_completion_webhook(
                    memory_request_batch,
                    batch_response,
                    end_user_id,
                    total_content_size,
                    total_storage_size
                )
            else:
                # Send webhook after background processing completes
                logger.info("Adding webhook as background task to wait for processing completion")
                background_tasks.add_task(
                    _send_batch_completion_webhook_after_background_processing,
                    memory_request_batch,
                    batch_response,
                    end_user_id,
                    total_content_size,
                    total_storage_size,
                    batch_id
                )
        else:
            logger.info("No webhook URL provided, skipping webhook notification")
        
        return batch_response

    except Exception as e:
        logger.error(f"Error processing batch: {e}", exc_info=True)
        return BatchMemoryResponse.failure(
            errors=[BatchMemoryError(index=-1, error=str(e))],
            code=500,
            error=str(e)
        )

async def _send_batch_completion_webhook(
    memory_request_batch: BatchMemoryRequest,
    batch_response: BatchMemoryResponse,
    user_id: str,
    total_content_size: int,
    total_storage_size: int
) -> None:
    """Send webhook notification for batch completion"""
    try:
        # Generate batch ID
        batch_id = str(uuid.uuid4())
        
        # Extract memory IDs from successful responses
        memory_ids = []
        for response in batch_response.successful:
            if response.data:
                for item in response.data:
                    memory_ids.append(item.memoryId)
        
        # Create webhook payload
        
        webhook_payload = webhook_service.create_batch_webhook_payload(
            batch_id=batch_id,
            user_id=user_id,
            status=batch_response.status,
            total_memories=batch_response.total_processed,
            successful_memories=batch_response.total_successful,
            failed_memories=batch_response.total_failed,
            errors=[{"index": error.index, "error": error.error} for error in batch_response.errors],
            memory_ids=memory_ids,
            processing_time_ms=int(time.time() * 1000)  # Simple timestamp for now
        )
        
        # Determine if we should use Azure Service Bus
        use_azure = bool(os.getenv("AZURE_SERVICE_BUS_CONNECTION_STRING"))
        # Send webhook
        success = await webhook_service.send_batch_completion_webhook(
            webhook_url=memory_request_batch.webhook_url,
            webhook_secret=memory_request_batch.webhook_secret,
            batch_data=webhook_payload,
            use_azure=use_azure
        )
        
        if success:
            logger.info(f"Webhook sent successfully for batch {batch_id}")
        else:
            logger.error(f"Failed to send webhook for batch {batch_id}")
        return success
    except Exception as e:
        logger.error(f"Error sending webhook notification: {e}", exc_info=True)
        return False    

async def _send_batch_completion_webhook_after_background_processing(
    memory_request_batch: BatchMemoryRequest,
    batch_response: BatchMemoryResponse,
    user_id: str,
    total_content_size: int,
    total_storage_size: int,
    batch_id: str,
    timeout_seconds: int = 60
) -> None:
    """
    Send webhook notification after background processing completes.
    This function waits for background tasks to finish before sending the webhook.
    """
    try:
        # Wait for background processing to complete with proper monitoring
        logger.info(f"Waiting for background processing to complete before sending webhook to: {memory_request_batch.webhook_url}")
        
        # Import the memory processing monitoring function
        from memory.memory_graph import wait_for_memory_processing_completion
        
        # Monitor memory processing background tasks until completion or timeout
        success = await wait_for_memory_processing_completion(
            batch_id=batch_id,
            timeout_seconds=timeout_seconds
        )
        
        if not success:
            logger.warning(f"Background processing did not complete successfully for batch {batch_id}, but sending webhook anyway")
        
        # Send the webhook notification
        success = await _send_batch_completion_webhook(
            memory_request_batch,
            batch_response,
            user_id,
            total_content_size,
            total_storage_size
        )
        if success:
            logger.info(f"Webhook notification sent successfully after background processing: {memory_request_batch.webhook_url}")
        else:
            logger.error(f"Failed to send webhook notification after background processing: {memory_request_batch.webhook_url}")
        return success
    except Exception as e:
        logger.error(f"Failed to send webhook notification after background processing: {e}")
        return False
        # Don't re-raise the exception as this is a background task
