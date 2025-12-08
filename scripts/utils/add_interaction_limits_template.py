"""
Template code for adding interaction limits check to route handlers.

Copy and paste this after authentication in your route handlers:
"""

# Standard template for 1 mini interaction operations
STANDARD_CHECK = """
        # Check interaction limits (1 mini interaction for {operation_name})
        from models.operation_types import MemoryOperationType
        from services.user_utils import User
        from config.features import get_features
        from os import environ as env
        
        features = get_features()
        if features.is_cloud and not env.get('EVALMETRICS', 'False').lower() == 'true':
            user_instance = User(id=user_id)
            limit_check = await user_instance.check_interaction_limits_fast(
                interaction_type='mini',
                memory_graph=memory_graph,
                operation=MemoryOperationType.{OPERATION_ENUM}
            )
            
            if limit_check:
                response_dict, status_code, is_error = limit_check
                if is_error:
                    response.status_code = status_code
                    return {ResponseType}.failure(
                        error=response_dict.get('error'),
                        code=status_code,
                        message=response_dict.get('message'),
                        details=response_dict
                    )
"""

# Mapping of routes to their operation enums
ROUTE_TO_OPERATION = {
    # Feedback routes
    "submit_feedback_v1": ("SUBMIT_FEEDBACK", "FeedbackResponse", "0 mini (user feedback collection)"),
    "submit_batch_feedback_v1": ("SUBMIT_BATCH_FEEDBACK", "BatchFeedbackResponse", "0 mini"),
    "get_feedback_by_id_v1": ("GET_FEEDBACK", "FeedbackResponse", "1 mini"),
    
    # GraphQL routes
    "graphql_proxy": ("GRAPHQL_QUERY", "Response", "1 mini (simple proxy)"),
    
    # Message routes  
    "store_message": ("STORE_MESSAGE", "MessageResponse", "1 mini"),
    "get_session_history": ("GET_SESSION_HISTORY", "SessionHistoryResponse", "1 mini"),
    "get_session_status": ("GET_SESSION_STATUS", "SessionStatusResponse", "1 mini"),
    "process_session_messages": ("PROCESS_SESSION_MESSAGES", "ProcessResponse", "1 mini"),
    
    # Schema routes
    "create_user_schema_v1": ("CREATE_SCHEMA", "SchemaResponse", "1 mini"),
    "list_user_schemas_v1": ("LIST_SCHEMAS", "SchemaListResponse", "1 mini"),
    "get_user_schema_v1": ("GET_SCHEMA", "SchemaResponse", "1 mini"),
    "update_user_schema_v1": ("UPDATE_SCHEMA", "SchemaResponse", "1 mini"),
    "delete_user_schema_v1": ("DELETE_SCHEMA", "Response", "1 mini"),
    
    # Sync routes
    "get_sync_tiers": ("GET_SYNC_TIERS", "SyncTiersResponse", "3 mini (embeddings + ranking)"),
    "get_sync_delta": ("GET_SYNC_DELTA", "Dict", "1 mini"),
}


def generate_check_code(operation_name: str) -> str:
    """Generate the check code for a given operation."""
    if operation_name not in ROUTE_TO_OPERATION:
        return f"# Operation {operation_name} not found in mapping"
    
    enum_name, response_type, description = ROUTE_TO_OPERATION[operation_name]
    
    code = f"""
        # Check interaction limits ({description})
        from models.operation_types import MemoryOperationType
        from services.user_utils import User
        from config.features import get_features
        from os import environ as env
        
        features = get_features()
        if features.is_cloud and not env.get('EVALMETRICS', 'False').lower() == 'true':
            user_instance = User(id=user_id)
            limit_check = await user_instance.check_interaction_limits_fast(
                interaction_type='mini',
                memory_graph=memory_graph,
                operation=MemoryOperationType.{enum_name}
            )
            
            if limit_check:
                response_dict, status_code, is_error = limit_check
                if is_error:
                    response.status_code = status_code
                    return {response_type}.failure(
                        error=response_dict.get('error'),
                        code=status_code,
                        message=response_dict.get('message')
                    )
"""
    return code


# Print all checks
if __name__ == "__main__":
    for operation in ROUTE_TO_OPERATION:
        print(f"\n{'='*80}")
        print(f"Operation: {operation}")
        print(f"{'='*80}")
        print(generate_check_code(operation))

