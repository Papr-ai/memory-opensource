"""
Memory Operation Types and their interaction costs.

This module defines all API operations that consume mini interactions
and their associated costs based on computational complexity.
"""

from enum import Enum
from typing import Optional


class MemoryOperationType(str, Enum):
    """
    Enum for all memory operation types across the API.
    
    Operations are categorized by their computational cost:
    - Simple operations (1 mini interaction): Basic CRUD, no LLM
    - Complex operations (multiple interactions): Use LLM, embeddings, graph processing
    """
    
    # Memory operations (memory_routes_v1.py)
    ADD_MEMORY = "add_memory_v1"  # 4 mini: goal prediction + search+rank + graph building
    ADD_MEMORY_BATCH = "add_memory_batch_v1"  # 4 * batch_size
    UPDATE_MEMORY = "update_memory_v1"  # 1 mini
    DELETE_MEMORY = "delete_memory_v1"  # 1 mini
    DELETE_ALL_MEMORIES = "delete_all_memories_v1"  # 1 mini
    GET_MEMORY = "get_memory_v1"  # 1 mini
    SEARCH_MEMORY = "search_v1"  # 1-3 mini (base + agentic + ranking)
    
    # Document operations (document_routes_v2.py)
    UPLOAD_DOCUMENT = "upload_document"  # 0 mini (charged via document ingestion fees)
    GET_DOCUMENT_STATUS = "get_document_status"  # 0 mini (status check only)
    CANCEL_DOCUMENT_PROCESSING = "cancel_document_processing"  # 0 mini
    
    # Feedback operations (feedback_routes.py)
    SUBMIT_FEEDBACK = "submit_feedback_v1"  # 0 mini (user feedback collection)
    SUBMIT_BATCH_FEEDBACK = "submit_batch_feedback_v1"  # 0 mini
    GET_FEEDBACK = "get_feedback_by_id_v1"  # 1 mini
    
    # GraphQL operations (graphql_routes.py)
    GRAPHQL_QUERY = "graphql_proxy"  # 1 mini (proxy only, no LLM)
    GRAPHQL_PLAYGROUND = "graphql_playground"  # 0 mini (static HTML)
    
    # Message operations (message_routes.py)
    STORE_MESSAGE = "store_message"  # 1 mini
    GET_SESSION_HISTORY = "get_session_history"  # 1 mini
    GET_SESSION_STATUS = "get_session_status"  # 1 mini
    PROCESS_SESSION_MESSAGES = "process_session_messages"  # 1 mini
    
    # Schema operations (schema_routes_v1.py)
    CREATE_SCHEMA = "create_user_schema_v1"  # 1 mini
    LIST_SCHEMAS = "list_user_schemas_v1"  # 1 mini
    GET_SCHEMA = "get_user_schema_v1"  # 1 mini
    UPDATE_SCHEMA = "update_user_schema_v1"  # 1 mini
    DELETE_SCHEMA = "delete_user_schema_v1"  # 1 mini
    
    # Sync operations (sync_routes.py)
    GET_SYNC_TIERS = "get_sync_tiers"  # 3 mini (embeddings + ranking)
    GET_SYNC_DELTA = "get_sync_delta"  # 1 mini
    SYNC_STREAM = "sync_stream"  # 1 mini


def get_operation_cost(
    operation: MemoryOperationType,
    batch_size: Optional[int] = None,
    enable_agentic_graph: bool = False,
    enable_rank_results: bool = False
) -> int:
    """
    Calculate the mini interaction cost for a given operation.
    
    Args:
        operation: The type of operation being performed
        batch_size: For batch operations, the number of items in the batch
        enable_agentic_graph: For search, whether agentic graph search is enabled
        enable_rank_results: For search, whether result ranking is enabled
        
    Returns:
        Number of mini interactions consumed by this operation
    """
    
    # Operations with 0 cost
    if operation in [
        MemoryOperationType.UPLOAD_DOCUMENT,
        MemoryOperationType.GET_DOCUMENT_STATUS,
        MemoryOperationType.CANCEL_DOCUMENT_PROCESSING,
        MemoryOperationType.SUBMIT_FEEDBACK,
        MemoryOperationType.SUBMIT_BATCH_FEEDBACK,
        MemoryOperationType.GRAPHQL_PLAYGROUND,
    ]:
        return 0
    
    # Search operation (variable cost based on features)
    if operation == MemoryOperationType.SEARCH_MEMORY:
        cost = 1  # Base search
        if enable_agentic_graph:
            cost += 1  # +1 for agentic graph reasoning
        if enable_rank_results:
            cost += 1  # +1 for result ranking
        return cost
    
    # Add memory operation (complex, multiple LLM calls)
    if operation == MemoryOperationType.ADD_MEMORY:
        return 4  # Goal prediction (1) + Search+Rank (2) + Graph building (1)
    
    # Batch add memory (scales with batch size)
    if operation == MemoryOperationType.ADD_MEMORY_BATCH:
        if batch_size is None:
            raise ValueError("batch_size required for ADD_MEMORY_BATCH operation")
        return 4 * batch_size  # Each memory costs 4 mini interactions
    
    # Sync tiers (complex, requires embeddings + ranking)
    if operation == MemoryOperationType.GET_SYNC_TIERS:
        return 3
    
    # All other operations cost 1 mini interaction
    return 1


# Operation to route mapping for logging
OPERATION_TO_ROUTE = {
    # Memory routes
    MemoryOperationType.ADD_MEMORY: "POST /v1/memory",
    MemoryOperationType.ADD_MEMORY_BATCH: "POST /v1/memory/batch",
    MemoryOperationType.UPDATE_MEMORY: "PUT /v1/memory/{id}",
    MemoryOperationType.DELETE_MEMORY: "DELETE /v1/memory/{id}",
    MemoryOperationType.DELETE_ALL_MEMORIES: "DELETE /v1/memory/all",
    MemoryOperationType.GET_MEMORY: "GET /v1/memory/{id}",
    MemoryOperationType.SEARCH_MEMORY: "POST /v1/memory/search",
    
    # Document routes
    MemoryOperationType.UPLOAD_DOCUMENT: "POST /v1/document",
    MemoryOperationType.GET_DOCUMENT_STATUS: "GET /v1/document/status/{id}",
    MemoryOperationType.CANCEL_DOCUMENT_PROCESSING: "DELETE /v1/document/{id}",
    
    # Feedback routes
    MemoryOperationType.SUBMIT_FEEDBACK: "POST /v1/feedback",
    MemoryOperationType.SUBMIT_BATCH_FEEDBACK: "POST /v1/feedback/batch",
    MemoryOperationType.GET_FEEDBACK: "GET /v1/feedback/{id}",
    
    # GraphQL routes
    MemoryOperationType.GRAPHQL_QUERY: "POST /v1/graphql",
    MemoryOperationType.GRAPHQL_PLAYGROUND: "GET /v1/graphql",
    
    # Message routes
    MemoryOperationType.STORE_MESSAGE: "POST /v1/messages",
    MemoryOperationType.GET_SESSION_HISTORY: "GET /v1/messages/history",
    MemoryOperationType.GET_SESSION_STATUS: "GET /v1/messages/status",
    MemoryOperationType.PROCESS_SESSION_MESSAGES: "POST /v1/messages/process",
    
    # Schema routes
    MemoryOperationType.CREATE_SCHEMA: "POST /v1/schemas",
    MemoryOperationType.LIST_SCHEMAS: "GET /v1/schemas",
    MemoryOperationType.GET_SCHEMA: "GET /v1/schemas/{id}",
    MemoryOperationType.UPDATE_SCHEMA: "PUT /v1/schemas/{id}",
    MemoryOperationType.DELETE_SCHEMA: "DELETE /v1/schemas/{id}",
    
    # Sync routes
    MemoryOperationType.GET_SYNC_TIERS: "POST /v1/sync/tiers",
    MemoryOperationType.GET_SYNC_DELTA: "GET /v1/sync/delta",
    MemoryOperationType.SYNC_STREAM: "WS /v1/sync/stream",
}

