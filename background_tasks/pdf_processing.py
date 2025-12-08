import logging
from memory.memory_graph import MemoryGraph, AsyncSession
from services.user_utils import User
from typing import Dict, Any, Optional, List
from fastapi import BackgroundTasks
import uuid
from memory.memory_item import DocumentMemoryItem
from models.parse_server import AddMemoryItem
from services.memory_service import handle_incoming_memory
from services.logger_singleton import LoggerSingleton
from models.memory_models import AddMemoryRequest

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)

async def add_page_to_memory_task(
    memory_request: AddMemoryRequest,
    user_id: str,
    session_token: str,
    neo_session: AsyncSession,
    memory_graph: MemoryGraph,
    background_tasks: BackgroundTasks,
    client_type: str = 'papr_plugin',
    user_workspace_ids: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    legacy_route: bool = True,
    workspace_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None,
    api_key_id: Optional[str] = None
) -> List[AddMemoryItem]:
    """Add a page to memory using the same path as regular memory additions"""
    try:

        # Use handle_incoming_memory to ensure consistent ACL and metadata handling
        response = await handle_incoming_memory(
            memory_request=memory_request,
            end_user_id=user_id,
            developer_user_id=user_id,
            sessionToken=session_token,
            neo_session=neo_session,
            user_info=None,  
            client_type=client_type,
            memory_graph=memory_graph,
            background_tasks=background_tasks,
            skip_background_processing=False,  
            user_workspace_ids=user_workspace_ids,
            api_key=api_key,
            legacy_route=True,
            workspace_id=workspace_id,
            organization_id=organization_id,
            namespace_id=namespace_id,
            api_key_id=api_key_id
        )

        if not response or not response.data:
            raise RuntimeError(f"Failed to add memory item for user {user_id}")

        return response.data

    except Exception as e:
        logger.error(f"Error in add_page_to_memory_task: {str(e)}", exc_info=True)
        raise 