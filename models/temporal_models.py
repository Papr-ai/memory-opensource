"""
Temporal-safe data models for workflows.

These models contain the essential types needed for Temporal workflows
without complex imports that could be restricted by the workflow sandbox.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from enum import Enum


class TemporalAddMemoryRequest(BaseModel):
    """Temporal-safe memory request for adding individual memories.
    
    This is a simplified version of the API AddMemoryRequest that only contains
    Temporal-safe fields. The conversion to API AddMemoryRequest happens in activities.
    """
    content: str = Field(..., description="The content of the memory to store")
    title: Optional[str] = Field(default=None, description="Optional title for the memory")
    type: Optional[str] = Field(default="text", description="Type of memory content")
    external_user_id: Optional[str] = Field(default=None, description="External user identifier")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class TemporalBatchMemoryRequest(BaseModel):
    """Temporal-safe request model for batch adding memories.
    
    This is a simplified version of the API BatchMemoryRequest for Temporal compatibility.
    The full graph_generation structure from the API is flattened here into individual fields.
    """
    user_id: Optional[str] = Field(default=None, description="Internal user ID")
    external_user_id: Optional[str] = Field(default=None, description="External user ID")
    organization_id: Optional[str] = Field(default=None, description="Organization ID for multi-tenant scoping")
    namespace_id: Optional[str] = Field(default=None, description="Namespace ID for multi-tenant scoping")
    schema_id: Optional[str] = Field(default=None, description="Schema ID for structured memory extraction (from graph_generation.auto.schema_id)")
    simple_schema_mode: Optional[bool] = Field(default=False, description="Use simplified schema mode (from graph_generation.auto.simple_schema_mode)")
    # Note: graph_override (manual mode) is handled via schema_specification dict in BatchWorkflowData
    memories: List[TemporalAddMemoryRequest] = Field(..., description="List of memory items to add")
    batch_size: Optional[int] = Field(default=10, description="Number of items to process in parallel")
    webhook_url: Optional[str] = Field(default=None, description="Optional webhook URL for completion notification")
    webhook_secret: Optional[str] = Field(default=None, description="Optional webhook secret")


class OptimizedAuthResponse(BaseModel):
    """Authentication response (Temporal-safe)"""
    developer_id: str
    end_user_id: str
    workspace_id: str
    organization_id: Optional[str] = None
    namespace_id: Optional[str] = None
    is_qwen_route: bool = False
    # Optional session token to allow activities that call Parse/Neo directly
    session_token: Optional[str] = None


class BatchWorkflowData(BaseModel):
    """Complete workflow data structure for Temporal"""
    batch_id: str
    batch_request: TemporalBatchMemoryRequest
    auth_response: OptimizedAuthResponse
    api_key: Optional[str] = None
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    schema_specification: Optional[Dict[str, Any]] = None  # SchemaSpecificationMixin data for graph extraction


class SchemaSpecification(BaseModel):
    """Temporal-safe schema specification to avoid heavy imports in workflows.
    
    Note: property_overrides uses Dict[str, Any] instead of PropertyOverrideRule to avoid
    circular imports. The actual PropertyOverrideRule validation happens in shared_types.py.
    """
    schema_id: Optional[str] = Field(default=None, description="Schema ID to enforce")
    simple_schema_mode: bool = Field(default=False, description="Use simplified schema mode")
    graph_override: Optional[Dict[str, Any]] = Field(default=None, description="Graph override structure")
    property_overrides: Optional[List[Dict[str, Any]]] = Field(default=None, description="Property overrides with match conditions (PropertyOverrideRule dicts)")


class DocumentFileReference(BaseModel):
    """File reference for large document processing"""
    upload_id: str
    file_url: str
    file_name: str
    file_size: int
    content_type: str
    parse_file_id: Optional[str] = None  # Parse Server file ID


class DocumentWorkflowData(BaseModel):
    """Document processing workflow data using file references"""
    upload_id: str
    file_reference: DocumentFileReference
    metadata: Dict[str, Any] = Field(default_factory=dict)
    organization_id: Optional[str] = None
    namespace_id: Optional[str] = None
    user_id: str
    workspace_id: Optional[str] = None
    preferred_provider: Optional[str] = None
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None


# Rebuild all models to ensure forward references are resolved
# This is needed for Temporal's Pydantic integration to properly serialize/deserialize workflow arguments
TemporalAddMemoryRequest.model_rebuild()
TemporalBatchMemoryRequest.model_rebuild()
OptimizedAuthResponse.model_rebuild()
BatchWorkflowData.model_rebuild()
SchemaSpecification.model_rebuild()
DocumentFileReference.model_rebuild()
DocumentWorkflowData.model_rebuild()

# Backward compatibility aliases (deprecated, use Temporal* versions)
AddMemoryRequest = TemporalAddMemoryRequest
BatchMemoryRequest = TemporalBatchMemoryRequest


def flatten_graph_generation_for_temporal(graph_generation: Optional[Any]) -> Dict[str, Any]:
    """
    Flatten the graph_generation structure from API models into a Temporal-safe dict.
    
    Converts all nested Pydantic models to plain dicts to ensure Temporal compatibility.
    
    Handles:
    - GraphGeneration (mode, auto, manual)
    - AutoGraphGeneration (schema_id, simple_schema_mode, property_overrides)
    - ManualGraphGeneration (nodes, relationships)
    - GraphOverrideNode (id, label, properties)
    - GraphOverrideRelationship (source_node_id, target_node_id, relationship_type, properties)
    - PropertyOverrideRule (match_conditions, overrides)
    
    Args:
        graph_generation: GraphGeneration object or dict from API request
        
    Returns:
        Flattened dict with all nested Pydantic models converted to plain dicts
    """
    if not graph_generation:
        return {}
    
    # Helper function to recursively convert Pydantic models to dicts
    def to_dict(obj):
        if obj is None:
            return None
        if hasattr(obj, 'model_dump'):
            return obj.model_dump(exclude_none=True)
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_dict(item) for item in obj]
        else:
            return obj
    
    # Convert the entire graph_generation structure to dict
    graph_gen_dict = to_dict(graph_generation)
    
    if not graph_gen_dict:
        return {}
    
    result = {}
    mode = graph_gen_dict.get('mode', 'auto')
    result['mode'] = mode
    
    if mode == 'auto':
        auto_config = graph_gen_dict.get('auto', {})
        if auto_config:
            # Flatten auto configuration fields
            if 'schema_id' in auto_config and auto_config['schema_id']:
                result['schema_id'] = auto_config['schema_id']
            if 'simple_schema_mode' in auto_config:
                result['simple_schema_mode'] = auto_config['simple_schema_mode']
            if 'property_overrides' in auto_config and auto_config['property_overrides']:
                # property_overrides is a list of PropertyOverrideRule dicts
                # Already converted to plain dicts by to_dict()
                result['property_overrides'] = auto_config['property_overrides']
                
    elif mode == 'manual':
        manual_config = graph_gen_dict.get('manual', {})
        if manual_config:
            # Convert manual mode to legacy graph_override format
            # nodes and relationships are already plain dicts from to_dict()
            result['graph_override'] = {
                'nodes': manual_config.get('nodes', []),
                'relationships': manual_config.get('relationships', [])
            }
    
    return result