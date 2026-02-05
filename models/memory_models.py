from pydantic import BaseModel, Field, ConfigDict, ValidationError, field_validator, RootModel
from typing import Dict, Any, List, Optional, Union, Literal
from models.shared_types import (
    MemoryMetadata, NodeLabel, ContextItem, MemoryType, MessageRole,
    UserMemoryCategory, AssistantMemoryCategory, PropertyOverrideRule,
    MemoryPolicy, PolicyMode, NodeConstraint, NodeSpec, RelationshipSpec,
    ConsentLevel, RiskLevel, ACLConfig, OMOFilter
)
from datetime import datetime, timezone, UTC
from memory.memory_item import MemoryItem
import json
from models.structured_outputs import (
    MemoryProperties, PersonProperties, CompanyProperties, 
    ProjectProperties, TaskProperties, InsightProperties, MeetingProperties, 
    OpportunityProperties, CodeProperties, LLMGraphNode, TaskPriority
)
from models.parse_server import ParseStoredMemory, Memory
from pydantic import model_validator
from enum import Enum
from os import environ as env
import uuid
from services.logger_singleton import LoggerSingleton


logger = LoggerSingleton.get_logger(__name__)
logger.info("Logger initialized at top of main.py!")


class ResponseFormat(str, Enum):
    """Response format options for API endpoints.
    
    - json: Standard JSON format (default)
    - toon: Token-Oriented Object Notation format for 30-60% token reduction in LLM contexts
    """
    JSON = "json"
    TOON = "toon"


class MemoryNodeProperties(BaseModel):
    """Properties specific to Memory nodes"""
    id: str
    type: str
    content: str
    memoryChunkIds: List[str]
    user_id: str
    workspace_id: Optional[str] = None
    pageId: Optional[str] = None
    title: Optional[str] = None
    topics: Optional[List[str]] = Field(default_factory=list)
    emotion_tags: Optional[List[str]] = Field(default_factory=list)
    emoji_tags: Optional[List[str]] = Field(default_factory=list)
    hierarchical_structures: Optional[Union[str, List]] = Field(default="")
    conversationId: Optional[str] = None
    sourceType: Optional[str] = None
    sourceUrl: Optional[str] = None
    # Multi-tenant scoping fields
    organization_id: Optional[str] = Field(default=None, description="Organization ID for multi-tenant scoping")
    namespace_id: Optional[str] = Field(default=None, description="Namespace ID for multi-tenant scoping")
    # New role and category fields for message-based memories
    memory_role: Optional[str] = Field(default=None, description="Role that generated this memory (user or assistant)")
    memory_category: Optional[str] = Field(default=None, description="Memory category based on role")
    user_read_access: List[str] = Field(default_factory=list)
    user_write_access: List[str] = Field(default_factory=list)
    workspace_read_access: List[str] = Field(default_factory=list)
    workspace_write_access: List[str] = Field(default_factory=list)
    role_read_access: List[str] = Field(default_factory=list)
    role_write_access: List[str] = Field(default_factory=list)
    external_user_read_access: Optional[List[str]] = Field(default_factory=list)
    external_user_write_access: Optional[List[str]] = Field(default_factory=list)
    namespace_read_access: Optional[List[str]] = Field(default_factory=list, description="Namespace-level read access")
    namespace_write_access: Optional[List[str]] = Field(default_factory=list, description="Namespace-level write access")
    organization_read_access: Optional[List[str]] = Field(default_factory=list, description="Organization-level read access")
    organization_write_access: Optional[List[str]] = Field(default_factory=list, description="Organization-level write access")
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None

    model_config = ConfigDict(extra='allow')

def memory_item_to_node(memory_item: 'MemoryItem', chunk_ids: List[str]) -> LLMGraphNode:
    """Convert a MemoryItem to a Node object for Neo4j storage"""
    # Extract metadata
    metadata = (json.loads(memory_item.metadata) 
               if isinstance(memory_item.metadata, str) 
               else memory_item.metadata)
    
    # Helper function to convert comma-separated string to list
    def string_to_list(value: Optional[Union[str, List[str]]]) -> List[str]:
        if not value:
            return []
        if isinstance(value, list):
            return value
        return [item.strip() for item in value.split(',')]

    # Merge customMetadata fields if present
    custom_metadata = metadata.get('customMetadata', {})
    custom_fields = custom_metadata if isinstance(custom_metadata, dict) else {}

    # Extract organization_id and namespace_id from metadata (top-level fields, not in customMetadata)
    organization_id = metadata.get('organization_id')
    namespace_id = metadata.get('namespace_id')

    # Create properties using Pydantic model for validation
    properties = MemoryNodeProperties(
        id=str(memory_item.id),
        type=memory_item.type,
        content=memory_item.content,
        memoryChunkIds=chunk_ids,
        user_id=metadata.get('user_id', ''),
        workspace_id=metadata.get('workspace_id'),
        pageId=metadata.get('pageId'),
        title=metadata.get('title'),
        # Convert string fields to lists
        topics=string_to_list(metadata.get('topics')),
        emotion_tags=string_to_list(metadata.get('emotion_tags',
                                  metadata.get('emotionTags',
                                  metadata.get('emotion tags', '')))),
        emoji_tags=string_to_list(metadata.get('emoji_tags',
                                  metadata.get('emojiTags',
                                  metadata.get('emoji tags', '')))),
        hierarchical_structures=(metadata.get('hierarchical_structures') or
                               metadata.get('hierarchicalStructures') or
                               metadata.get('hierarchical structures') or
                               ''),
        conversationId=metadata.get('conversationId'),
        sourceType=metadata.get('sourceType'),
        sourceUrl=metadata.get('sourceUrl'),
        # Multi-tenant scoping fields
        organization_id=organization_id,
        namespace_id=namespace_id,
        # Extract role and category from primary metadata fields
        memory_role=metadata.get('role'),
        memory_category=metadata.get('category'),
        user_read_access=metadata.get('user_read_access', []),
        user_write_access=metadata.get('user_write_access', []),
        workspace_read_access=metadata.get('workspace_read_access', []),
        workspace_write_access=metadata.get('workspace_write_access', []),
        role_read_access=metadata.get('role_read_access', []),
        role_write_access=metadata.get('role_write_access', []),
        external_user_read_access=metadata.get('external_user_read_access', []),
        external_user_write_access=metadata.get('external_user_write_access', []),
        namespace_read_access=metadata.get('namespace_read_access', []),
        namespace_write_access=metadata.get('namespace_write_access', []),
        organization_read_access=metadata.get('organization_read_access', []),
        organization_write_access=metadata.get('organization_write_access', []),
        createdAt=metadata.get('createdAt') or datetime.now(timezone.utc).isoformat(),
        **custom_fields
    ).model_dump(exclude_none=True)

    # Create Node object
    return LLMGraphNode(
        label="Memory",
        properties=properties
    )

class NeoBaseProperties(BaseModel):
    """Base properties that all Neo4j nodes should inherit from"""
    model_config = ConfigDict(
        extra='forbid'
    )
    
    # Optional ACL properties
    user_read_access: Optional[List[str]] = Field(default_factory=list)
    user_write_access: Optional[List[str]] = Field(default_factory=list)
    workspace_read_access: Optional[List[str]] = Field(default_factory=list)
    workspace_write_access: Optional[List[str]] = Field(default_factory=list)
    role_read_access: Optional[List[str]] = Field(default_factory=list)
    role_write_access: Optional[List[str]] = Field(default_factory=list)
    external_user_read_access: Optional[List[str]] = Field(default_factory=list)
    external_user_write_access: Optional[List[str]] = Field(default_factory=list)
    namespace_read_access: Optional[List[str]] = Field(default_factory=list, description="Namespace-level read access")
    namespace_write_access: Optional[List[str]] = Field(default_factory=list, description="Namespace-level write access")
    organization_read_access: Optional[List[str]] = Field(default_factory=list, description="Organization-level read access")
    organization_write_access: Optional[List[str]] = Field(default_factory=list, description="Organization-level write access")

     # Multi-tenant scoping fields
    organization_id: Optional[str] = Field(default=None, description="Organization ID for multi-tenant scoping")
    namespace_id: Optional[str] = Field(default=None, description="Namespace ID for multi-tenant scoping")
    
    # Optional metadata fields
    workspace_id: Optional[str] = None
    user_id: Optional[str] = None
    external_user_id: Optional[str] = None
    sourceType: Optional[str] = None
    sourceUrl: Optional[str] = None
    pageId: Optional[str] = None
    conversationId: Optional[str] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def sync_created_at_fields(cls, data):
        if isinstance(data, dict):
            # Handle createdAt/created_at for datetime
            created = data.get('createdAt') or data.get('created_at')
            if created:
                if isinstance(created, str):
                    try:
                        created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    except Exception:
                        created_dt = None
                elif isinstance(created, datetime):
                    created_dt = created
                else:
                    created_dt = None
                data['createdAt'] = created if isinstance(created, str) else created_dt.isoformat() if created_dt else None
                # Do NOT set data['created_at']
            # Same for updatedAt
            updated = data.get('updatedAt') or data.get('updated_at')
            if updated:
                if isinstance(updated, str):
                    try:
                        updated_dt = datetime.fromisoformat(updated.replace('Z', '+00:00'))
                    except Exception:
                        updated_dt = None
                elif isinstance(updated, datetime):
                    updated_dt = updated
                else:
                    updated_dt = None
                data['updatedAt'] = updated if isinstance(updated, str) else updated_dt.isoformat() if updated_dt else None
                # Do NOT set data['updated_at']
        return data

    @model_validator(mode="before")
    @classmethod
    def fix_created_at_fields(cls, data):
        if isinstance(data, dict):
            for field in ["created_at", "updated_at", "createdAt", "updatedAt"]:
                value = data.get(field)
                if isinstance(value, str):
                    if value.lower() in ("none", "null", ""):
                        data[field] = None
                    else:
                        try:
                            data[field] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                        except Exception:
                            pass  # Let Pydantic handle invalid formats
        return data



class BaseNodeProperties(BaseModel):
    """Base properties for all node types"""
    # Optional ACL properties
    user_read_access: Optional[List[str]] = Field(default_factory=list)
    user_write_access: Optional[List[str]] = Field(default_factory=list)
    workspace_read_access: Optional[List[str]] = Field(default_factory=list)
    workspace_write_access: Optional[List[str]] = Field(default_factory=list)
    role_read_access: Optional[List[str]] = Field(default_factory=list)
    role_write_access: Optional[List[str]] = Field(default_factory=list)
    external_user_read_access: Optional[List[str]] = Field(default_factory=list)
    external_user_write_access: Optional[List[str]] = Field(default_factory=list)
    namespace_read_access: Optional[List[str]] = Field(default_factory=list, description="Namespace-level read access")
    namespace_write_access: Optional[List[str]] = Field(default_factory=list, description="Namespace-level write access")
    organization_read_access: Optional[List[str]] = Field(default_factory=list, description="Organization-level read access")
    organization_write_access: Optional[List[str]] = Field(default_factory=list, description="Organization-level write access")

    # Multi-tenant scoping fields
    organization_id: Optional[str] = Field(default=None, description="Organization ID for multi-tenant scoping")
    namespace_id: Optional[str] = Field(default=None, description="Namespace ID for multi-tenant scoping")
    
    # Optional metadata fields
    workspace_id: Optional[str] = None
    user_id: Optional[str] = None
    source_type: Optional[str] = Field(None, alias='sourceType')
    source_url: Optional[str] = Field(None, alias='sourceUrl')
    page_id: Optional[str] = Field(None, alias='pageId')
    conversation_id: Optional[str] = Field(None, alias='conversationId')
    created_at: Optional[datetime] = Field(None, alias='createdAt')
    updated_at: Optional[datetime] = Field(None, alias='updatedAt')


    model_config = ConfigDict(
        populate_by_name=True,  # Allow both field name and alias
        extra='allow'  # Allow extra fields to be passed
    )

class NeoMemoryNode(MemoryProperties, NeoBaseProperties):
    """Memory node with all Neo4j properties"""
    title: Optional[str] = None
    emoji_tags: Optional[List[str]] = Field(default_factory=list)
    hierarchical_structures: Optional[Union[str, List]] = Field(default="")

    pass

class PaprMemoryNodeProperties(MemoryProperties, BaseNodeProperties):
    """Memory node properties"""
    title: Optional[str] = None
    emoji_tags: Optional[List[str]] = Field(default_factory=list)
    hierarchical_structures: Optional[Union[str, List]] = Field(default="")
    steps: List[str] = Field(default_factory=list)  # Default to empty list
    current_step: str = Field(default="")  # Default to empty string

    pass

class NeoPersonNode(PersonProperties, NeoBaseProperties):
    """Person node with all Neo4j properties"""
    pass

class PersonNodeProperties(PersonProperties, BaseNodeProperties):
    """Person node properties"""
    pass

class NeoCompanyNode(CompanyProperties, NeoBaseProperties):
    """Company node with all Neo4j properties"""
    pass

class CompanyNodeProperties(CompanyProperties, BaseNodeProperties):
    """Company node properties"""
    pass

class NeoProjectNode(ProjectProperties, NeoBaseProperties):
    """Project node with all Neo4j properties"""
    pass

class ProjectNodeProperties(ProjectProperties, BaseNodeProperties):
    """Project node properties"""
    pass

class NeoTaskNode(TaskProperties, NeoBaseProperties):
    """Task node with all Neo4j properties"""
    pass

class TaskNodeProperties(TaskProperties, BaseNodeProperties):
    """Task node properties"""
    pass

class NeoInsightNode(InsightProperties, NeoBaseProperties):
    """Insight node with all Neo4j properties"""
    pass

class InsightNodeProperties(InsightProperties, BaseNodeProperties):
    """Insight node properties"""
    pass

class NeoMeetingNode(MeetingProperties, NeoBaseProperties):
    """Meeting node with all Neo4j properties"""
    pass

class MeetingNodeProperties(MeetingProperties, BaseNodeProperties):
    """Meeting node properties"""
    pass

class NeoOpportunityNode(OpportunityProperties, NeoBaseProperties):
    """Opportunity node with all Neo4j properties"""
    pass

class OpportunityNodeProperties(OpportunityProperties, BaseNodeProperties):
    """Opportunity node properties"""
    pass

class NeoCodeNode(CodeProperties, NeoBaseProperties):
    """Code node with all Neo4j properties"""
    pass

class FlexibleCustomNode(BaseModel):
    """Flexible node container for custom node types that accepts any fields"""
    model_config = ConfigDict(extra='allow')  # Allow extra fields
    
    # Minimal required fields
    id: str
    
    def __init__(self, **data):
        # Ensure id is present
        if 'id' not in data:
            data['id'] = ''
        super().__init__(**data)

class CustomNodeProperties(BaseModel):
    """Type-safe properties class for custom node types with guaranteed JSON serialization"""
    model_config = ConfigDict(extra='forbid')  # Enforce schema for API consistency
    
    # Required fields for all custom nodes
    id: str
    
    # Common optional fields that custom nodes might have
    name: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    version: Optional[str] = None
    email: Optional[str] = None
    
    # Additional metadata (JSON-serializable only)
    metadata: Dict[str, Union[str, int, float, bool, None]] = Field(default_factory=dict)
    
    @classmethod
    def from_flexible_node(cls, flexible_node: FlexibleCustomNode) -> 'CustomNodeProperties':
        """Convert a flexible node to type-safe properties, ensuring JSON serialization"""
        if hasattr(flexible_node, 'model_dump'):
            data = flexible_node.model_dump()
        else:
            data = dict(flexible_node)
        
        # Extract known fields
        known_fields = {
            'id': data.get('id', ''),
            'name': data.get('name'),
            'description': data.get('description'), 
            'language': data.get('language'),
            'version': data.get('version'),
            'email': data.get('email'),
        }
        
        # Put remaining fields in metadata (with JSON serialization safety)
        metadata = {}
        for key, value in data.items():
            if key not in known_fields and value is not None:
                # Ensure JSON serializable
                if isinstance(value, (str, int, float, bool, type(None))):
                    metadata[key] = value
                else:
                    # Convert non-serializable types to strings
                    metadata[key] = str(value)
        
        known_fields['metadata'] = metadata
        
        # Remove None values to keep response clean
        return cls(**{k: v for k, v in known_fields.items() if v is not None})

class CodeNodeProperties(CodeProperties, BaseNodeProperties):
    """Code node properties"""
    pass

class NeoNode(BaseModel):
    """Generic Neo4j node that combines label and type-specific properties"""
    label: NodeLabel
    properties: Union[
        NeoMemoryNode,
        NeoPersonNode,
        NeoCompanyNode,
        NeoProjectNode,
        NeoTaskNode,
        NeoInsightNode,
        NeoMeetingNode,
        NeoOpportunityNode,
        NeoCodeNode,
        FlexibleCustomNode  # Add support for custom nodes
    ]

class Node(BaseModel):
    """Public-facing node structure - supports both system and custom schema nodes"""
    label: str = Field(..., description="Node type label - can be system type (Memory, Person, etc.) or custom type from UserGraphSchema")
    properties: Dict[str, Any] = Field(..., description="Node properties - structure depends on node type and schema")
    schema_id: Optional[str] = Field(
        None, 
        description="Reference to UserGraphSchema ID for custom nodes. Use GET /v1/schemas/{schema_id} to get full schema definition. Null for system nodes."
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "System Node Example",
                    "summary": "Standard system node (Person)",
                    "value": {
                        "label": "Person",
                        "properties": {
                            "id": "person-123",
                            "name": "John Doe",
                            "role": "Manager",
                            "user_id": "user-456"
                        },
                        "schema_id": None
                    }
                },
                {
                    "name": "Custom Node Example",
                    "summary": "Custom schema node (Developer)",
                    "value": {
                        "label": "Developer",
                        "properties": {
                            "id": "dev-789",
                            "name": "Rachel Green",
                            "expertise": ["React", "TypeScript"],
                            "years_experience": 5,
                            "user_id": "user-456"
                        },
                        "schema_id": "schema_abc123"
                    }
                }
            ]
        }
    )

    @classmethod
    def from_internal(cls, neo_node: NeoNode, schema_id: Optional[str] = None) -> 'Node':
        """Convert internal NeoNode to public Node"""
        # Convert properties to dict format
        properties_data = neo_node.properties.model_dump()
        
        # Map internal property names to external ones
        common_mappings = {
            'createdAt': 'created_at',
            'updatedAt': 'updated_at',
            'sourceType': 'source_type',
            'sourceUrl': 'source_url',
            'emojiTags': 'tags',
        }

        # Update property names
        for internal, external in common_mappings.items():
            if internal in properties_data:
                properties_data[external] = properties_data.pop(internal)

        # Get label as string (handle both enum and string types)
        label_str = neo_node.label.value if hasattr(neo_node.label, 'value') else str(neo_node.label)
        
        # Determine if this is a custom node (schema_id provided or not a system label)
        from models.shared_types import NodeLabel
        is_system_node = NodeLabel.is_system_label(label_str)
        
        # Set schema_id for custom nodes
        final_schema_id = None if is_system_node else schema_id
        
        return cls(
            label=label_str,
            properties=properties_data,
            schema_id=final_schema_id
        )

class MemorySourceLocation(BaseModel):
    """
    Tracks presence of a memory item in different storage systems.
    """
    in_qdrant: bool = Field(default=False, description="Memory exists in Qdrant")
    in_qdrant_grouped: bool = Field(default=False, description="Memory exists in Qdrant as grouped memory")
    in_neo: bool = Field(default=False, description="Memory exists in Neo4j")

class MemoryIDSourceLocation(BaseModel):
    memory_id: str
    source_location: MemorySourceLocation

class MemorySourceInfo(BaseModel):
    memory_id_source_location: List[MemoryIDSourceLocation]


class RerankingProvider(str, Enum):
    """Reranking provider options for memory search results.

    OPENAI: LLM-based reranking using GPT models. Returns score + confidence.
            Models: gpt-5-nano (fast), gpt-5-mini (better quality), gpt-4o-mini
    COHERE: Cross-encoder reranking. Faster, optimized for relevance ranking.
            Models: rerank-v3.5 (latest), rerank-english-v3.0, rerank-multilingual-v3.0
    """
    OPENAI = "openai"      # LLM-based (gpt-5-nano default, gpt-5-mini for quality)
    COHERE = "cohere"      # Cross-encoder (rerank-v3.5 default)


class RerankingConfig(BaseModel):
    """Configuration for reranking memory search results"""
    reranking_enabled: bool = Field(
        default=False,
        description="Whether to enable reranking of search results"
    )
    reranking_provider: RerankingProvider = Field(
        default=RerankingProvider.OPENAI,
        description="Reranking provider: 'openai' (better quality, slower) or 'cohere' (faster, optimized for reranking)"
    )
    reranking_model: str = Field(
        default="gpt-5-nano",
        description="Model to use for reranking. OpenAI (LLM): 'gpt-5-nano' (fast reasoning, default), 'gpt-5-mini' (better quality reasoning). Cohere (cross-encoder): 'rerank-v3.5' (latest), 'rerank-english-v3.0', 'rerank-multilingual-v3.0'"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "reranking_enabled": True,
                    "reranking_provider": "openai",
                    "reranking_model": "gpt-5-nano"
                },
                {
                    "reranking_enabled": True,
                    "reranking_provider": "openai",
                    "reranking_model": "gpt-5-mini"
                },
                {
                    "reranking_enabled": True,
                    "reranking_provider": "cohere",
                    "reranking_model": "rerank-v3.5"
                }
            ]
        }
    )


class RelatedMemoryResult(BaseModel):
    """Return type for find_related_memory_items_async"""
    memory_items: List[ParseStoredMemory]
    neo_nodes: List[NeoNode]
    neo_context: Optional[str] = None
    neo_query: Optional[str] = None
    memory_source_info: Optional[MemorySourceInfo] = None
    confidence_scores: Optional[List[float]] = Field(default_factory=list, description="Reranker scores for memory items (parallel to memory_items). For LLM: normalized 1-10 score to 0-1. For cross-encoder: raw relevance score.")
    llm_confidence_scores: Optional[List[float]] = Field(default_factory=list, description="LLM confidence scores (0-1) for LLM reranking only. Represents LLM's confidence in its relevance judgment.")
    similarity_scores_by_id: Optional[Dict[str, float]] = Field(default_factory=dict, description="Cosine similarity scores for each memory id")
    bigbird_memory_info: Optional[Any] = None

    def log_summary(self) -> None:
        """Log a summary of the results"""
        logger.info(f"Found {len(self.memory_items)} memory items")
        logger.info(f"Found {len(self.neo_nodes)} neo nodes")
        logger.info(f"Found {len(self.memory_source_info.memory_id_source_location)} source locations")
        if self.confidence_scores:
            logger.info(f"Confidence scores: {self.confidence_scores[:5]}...")  # Log first 5 scores
        if self.neo_nodes:
            logger.info("Sample of neo nodes (first 3):")
            for i, node in enumerate(self.neo_nodes[:3]):
                logger.info(f"Node {i + 1}:")
                logger.info(f"  Label: {node.label}")
                logger.info(f"  Properties: {json.dumps(node.properties.model_dump(), indent=2)}")
        logger.info(f"Neo context: {self.neo_context}")
        logger.info(f"Neo query: {self.neo_query}")

class SearchResult(BaseModel):
    """Return type for SearchResult"""
    memories: List[Memory]
    nodes: List[Node]
    schemas_used: Optional[List[str]] = Field(
        None,
        description="List of UserGraphSchema IDs used in this response. Use GET /v1/schemas/{id} to get full schema definitions."
    )

    @classmethod
    def from_internal(cls, internal_result: RelatedMemoryResult, schema_mapping: Optional[Dict[str, str]] = None) -> 'SearchResult':
        """Convert internal result to public-facing result"""
        memories = [
            Memory.from_internal(item) 
            for item in (internal_result.memory_items or [])
            if item is not None
        ]
        
        # Convert nodes with schema information
        nodes = []
        used_schemas = set()
        
        for node in (internal_result.neo_nodes or []):
            if node is not None:
                # Get schema_id for this node if it's a custom node
                node_label = node.label.value if hasattr(node.label, 'value') else str(node.label)
                schema_id = schema_mapping.get(node_label) if schema_mapping else None
                if not schema_id:
                    # Fall back to schema_id stored on the node itself
                    schema_id = getattr(getattr(node, "properties", None), "schema_id", None)

                # Create the public node
                public_node = Node.from_internal(node, schema_id=schema_id)
                nodes.append(public_node)

                # Track used schemas
                if schema_id:
                    used_schemas.add(schema_id)
        
        # Convert set to sorted list for consistent output
        schemas_used_list = sorted(list(used_schemas)) if used_schemas else None
        
        return cls(
            memories=memories, 
            nodes=nodes,
            schemas_used=schemas_used_list
        )


class GetMemoryResponse(BaseModel):
    data: RelatedMemoryResult
    success: bool = True
    error: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data": [],
                "success": True,
                "error": None
            }
        }
    )

class SearchOverridePattern(BaseModel):
    """Developer-specified search pattern for search override"""
    source_label: str = Field(
        ..., 
        description="Source node label (e.g., 'Memory', 'Person', 'Company'). Must match schema node types."
    )
    relationship_type: str = Field(
        ..., 
        description="Relationship type (e.g., 'ASSOCIATED_WITH', 'WORKS_FOR'). Must match schema relationship types."
    )
    target_label: str = Field(
        ..., 
        description="Target node label (e.g., 'Person', 'Company', 'Project'). Must match schema node types."
    )
    direction: str = Field(
        default="->", 
        description="Relationship direction: '->' (outgoing), '<-' (incoming), or '-' (bidirectional)"
    )
    
    model_config = ConfigDict(extra='forbid')

class SearchOverrideFilter(BaseModel):
    """Property filters for search override"""
    node_type: str = Field(
        ..., 
        description="Node type to filter (e.g., 'Person', 'Memory', 'Company')"
    )
    property_name: str = Field(
        ..., 
        description="Property name to filter on (e.g., 'name', 'content', 'role')"
    )
    operator: str = Field(
        ..., 
        description="Filter operator: 'CONTAINS', 'EQUALS', 'STARTS_WITH', 'IN'"
    )
    value: Union[str, List[str], int, float, bool] = Field(
        ..., 
        description="Filter value(s). Use list for 'IN' operator."
    )
    
    model_config = ConfigDict(extra='forbid')

class SearchOverrideSpecification(BaseModel):
    """Complete search override specification provided by developer"""
    pattern: SearchOverridePattern = Field(
        ..., 
        description="Graph pattern to search for (source)-[relationship]->(target)"
    )
    filters: List[SearchOverrideFilter] = Field(
        default=[], 
        description="Property filters to apply to the search pattern"
    )
    return_properties: Optional[List[str]] = Field(
        default=None, 
        description="Specific properties to return. If not specified, returns all properties."
    )
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "example": {
                "pattern": {
                    "source_label": "Memory",
                    "relationship_type": "ASSOCIATED_WITH", 
                    "target_label": "Person",
                    "direction": "->"
                },
                "filters": [
                    {
                        "node_type": "Person",
                        "property_name": "name",
                        "operator": "CONTAINS",
                        "value": "John"
                    },
                    {
                        "node_type": "Memory",
                        "property_name": "topics",
                        "operator": "IN",
                        "value": ["project", "meeting"]
                    }
                ],
                "return_properties": ["name", "content", "createdAt"]
            }
        }
    )


class SearchRequest(BaseModel):
    """Search request parameters"""
    query: str = Field(
        ...,  # Makes it required
        description=(
            "Detailed search query describing what you're looking for. For best results, write 2-3 "
            "sentences that include specific details, context, and time frame. Examples: "
            "'Find recurring customer complaints about API performance from the last month. Focus on "
            "issues where customers specifically mentioned timeout errors or slow response times in "
            "their conversations.' "
            "'What are the main issues and blockers in my current projects? Focus on technical challenges and timeline impacts.' "
            "'Find insights about team collaboration and communication patterns from recent meetings and discussions.'"
        ),
        examples=[
            "Show me the most critical customer pain points from my recent conversations. Focus on "
            "issues that multiple customers have mentioned and any specific feature requests or "
            "workflow improvements they've suggested.",
            "What are the main issues and blockers in my current projects? Focus on technical challenges and timeline impacts.",
            "Find insights about team collaboration and communication patterns from recent meetings and discussions."
        ]
    )
    rank_results: bool = Field(
        default=False,
        description=(
            "DEPRECATED: Use 'reranking_config' instead. "
            "Whether to enable additional ranking of search results. Default is false because results "
            "are already ranked when using an LLM for search (recommended approach). Only enable this "
            "if you're not using an LLM in your search pipeline and need additional result ranking. "
            "Migration: Replace 'rank_results: true' with 'reranking_config: {reranking_enabled: true, "
            "reranking_provider: \"cohere\", reranking_model: \"rerank-v3.5\"}'"
        ),
        json_schema_extra={"deprecated": True}
    )
    enable_agentic_graph: bool = Field(
        default=False,
        description=(
            "HIGHLY RECOMMENDED: Enable agentic graph search for intelligent, context-aware results. "
            "When enabled, the system can understand ambiguous references by first identifying specific entities "
            "from your memory graph, then performing targeted searches. Examples: "
            "'customer feedback' → identifies your customers first, then finds their specific feedback; "
            "'project issues' → identifies your projects first, then finds related issues; "
            "'team meeting notes' → identifies team members first, then finds meeting notes. "
            "This provides much more relevant and comprehensive results. "
            "Set to false only if you need faster, simpler keyword-based search."
        )
    )
    # User identification (external_user_id is primary)
    external_user_id: Optional[str] = Field(
        default=None,
        description="Your application's user identifier to filter search results. This is the primary way to identify users. "
                   "Use this for your app's user IDs (e.g., 'user_alice_123', UUID, email)."
    )
    # Deprecated field (kept for backwards compatibility)
    user_id: Optional[str] = Field(
        default=None,
        description="DEPRECATED: Use 'external_user_id' instead. Internal Papr Parse user ID. "
                   "Most developers should not use this field directly.",
        json_schema_extra={"deprecated": True}
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Optional organization ID for multi-tenant search scoping. When provided, search is scoped to memories within this organization."
    )
    namespace_id: Optional[str] = Field(
        default=None,
        description="Optional namespace ID for multi-tenant search scoping. When provided, search is scoped to memories within this namespace."
    )
    schema_id: Optional[str] = Field(
        default=None,
        description="Optional user-defined schema ID to use for this search. If provided, this schema (plus system schema) will be used for query generation. If not provided, system will automatically select relevant schema based on query content."
    )
    metadata: Optional[MemoryMetadata] = Field(
        default=None,
        description="Optional metadata filter. Any field in MemoryMetadata (including custom fields) can be used for filtering."
    )
    search_override: Optional[SearchOverrideSpecification] = Field(
        None,
        description="**OPTIONAL**: Override automatic search query generation with your own exact graph pattern and filters. **⚡ AUTOMATIC BY DEFAULT**: If not provided, the system automatically generates optimized Cypher queries using AI - no action required! **🎯 USE WHEN**: You want precise control over search patterns, have specific graph traversals in mind, or want to bypass AI query generation for performance. **📋 VALIDATION**: All patterns and filters must comply with your schema definitions."
    )
    reranking_config: Optional[RerankingConfig] = Field(
        default=None,
        description="Optional reranking configuration. If provided, enables reranking with specified provider (OpenAI or Cohere) and model. If not provided but rank_results=True, uses default OpenAI reranking."
    )

    # OMO (Open Memory Object) Filtering - filter search results by safety standards
    omo_filter: Optional[OMOFilter] = Field(
        default=None,
        description="Optional OMO (Open Memory Object) safety filter. Filter search results by consent level and/or risk level. "
                   "Use this to exclude memories without proper consent or flagged content from search results."
    )

    @model_validator(mode='after')
    def resolve_reranking_config(self) -> 'SearchRequest':
        """
        Resolve conflicts between deprecated rank_results and new reranking_config.

        Precedence:
        1. If only reranking_config is set → use it
        2. If only rank_results is set → use it (with deprecation warning logged)
        3. If both are set → reranking_config takes precedence
        4. If both are set with conflicting values → log warning, use reranking_config
        """
        import warnings

        # Check if rank_results was explicitly set (not default)
        rank_results_set = self.rank_results is True
        reranking_config_set = self.reranking_config is not None

        if rank_results_set and not reranking_config_set:
            # Case: Only rank_results is set - emit deprecation warning
            warnings.warn(
                "The 'rank_results' field is deprecated. "
                "Use 'reranking_config' instead. Example: "
                "reranking_config={'reranking_enabled': True, 'reranking_provider': 'cohere', 'reranking_model': 'rerank-v3.5'}",
                DeprecationWarning,
                stacklevel=2
            )
        elif rank_results_set and reranking_config_set:
            # Case: Both are set - check for conflicts
            reranking_enabled = self.reranking_config.reranking_enabled if self.reranking_config else False

            if self.rank_results != reranking_enabled:
                # Conflict: rank_results=True but reranking_config.reranking_enabled=False (or vice versa)
                logger.warning(
                    f"Conflict between deprecated 'rank_results' ({self.rank_results}) and "
                    f"'reranking_config.reranking_enabled' ({reranking_enabled}). "
                    f"Using 'reranking_config' value. Please remove 'rank_results' from your request."
                )

        return self

    @field_validator('query')
    @classmethod
    def validate_and_transform_query(cls, v: str) -> str:
        """Transform generic search queries into more specific ones"""
        v = v.strip()
        
        # List of generic queries to transform
        generic_queries = {
            "*": "Show me my most recent memories from the past few weeks.",
            "all": "Find my most recent memories from the past few weeks.",
            "search all": "Retrieve my most recent memories from the past few weeks.",
            "everything": "Show me my most recent memories from the past few weeks.",
            "search all memories": "Find my most recent memories from the past few weeks.",
            "all memories": "Show me my most recent memories from the past few weeks."
        }

        # Transform generic queries into more specific ones
        if v.lower() in generic_queries:
            return generic_queries[v.lower()]

        # Check for partial matches
        if any(phrase in v for phrase in ["search all", "all memories", "show all"]):
            return "Show me my most recent memories from the past few weeks."
        
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Find recurring customer complaints about API performance from the last month. "
                        "Focus on issues that multiple customers have mentioned and any specific feature requests or "
                        "workflow improvements they've suggested.",
                "rank_results": True,
                "enable_agentic_graph": False,
                "external_user_id": "external_user_123"
                },
                # OPTION 1: Automatic search (most common - just omit search_override)
                # The system will automatically generate optimized Cypher queries using AI
                
                # OPTION 2: Override with your own search pattern (advanced use case)
                "search_override": {
                    "pattern": {
                        "source_label": "Memory",
                        "relationship_type": "ASSOCIATED_WITH",
                        "target_label": "Person",
                        "direction": "->"
                    },
                    "filters": [
                        {
                            "node_type": "Person",
                            "property_name": "role",
                            "operator": "CONTAINS",
                            "value": "customer"
                        },
                        {
                            "node_type": "Memory",
                            "property_name": "topics",
                            "operator": "IN",
                            "value": ["API", "performance", "complaints"]
                        }
                    ],
                    "return_properties": ["name", "content", "createdAt", "topics"]
                
            }
        }
    )

class ImageGenerationCategory(str, Enum):
    """Categories for memory items to specify their nature and content"""
    NARRATIVE_ELEMENT = "narrative_element"
    RPG_ACTION = "rpg_action"
    OBJECT_DESCRIPTION = "object_description"
    ART_IDEA = "art_idea"
    DREAM_OR_FANTASY = "dream_or_fantasy"
    HISTORICAL_EVENT = "historical_event"
    PERSONAL_MEMORY = "personal_memory"
    BIOLOGICAL_CONCEPT = "biological_concept"
    CULTURAL_REFERENCE = "cultural_reference"
    TRAVEL = "travel"
    MOOD_OR_EMOTION = "mood_or_emotion"
    WEB_DEVELOPMENT = "web_development"
    TECHNICAL_LOG = "technical_log"
    CODE_SNIPPET = "code_snippet"
    ERROR_MESSAGE = "error_message"
    BUSINESS_MANAGEMENT = "business_management"
    PROJECT_MANAGEMENT = "project_management"
    DOCUMENT = "document"
    INSTRUCTION = "instruction"
    STRUCTURED_LIST = "structured_list"
    COMMUNICATION = "communication"
    MEDICAL = "medical"
    TECHNICAL_ANALYSIS = "technical_analysis"
    RESUME = "resume"
    PERSONAL_IDENTIFIERS = "personal_identifiers"
    AMBIGUOUS_CONCEPT = "ambiguous_concept"
    ART_RELATED = "art_related"
    PRODUCT_IDEA = "product_idea"
    CALENDAR_EVENT = "calendar_event"
    OTHER = "other"



class RelationshipType(str, Enum):
    """Enum for relationship types"""
    PREVIOUS_MEMORY_ITEM = "previous_memory_item_id"
    ALL_PREVIOUS_MEMORY_ITEMS = "all_previous_memory_items"
    LINK_TO_ID = "link_to_id"  # For linking to a specific memory ID

class RelationshipItem(BaseModel):
    """Relationship item for memory request"""
    relation_type: str
    # Either a specific memory ID or one of the special relationship types
    related_item_id: Optional[str] = None
    # Enum for special relationship types
    relationship_type: Optional[RelationshipType] = None
    # Legacy field - not used in processing but kept for backward compatibility
    related_item_type: Optional[str] = Field(default="Memory", description="Legacy field - not used in processing")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('related_item_id', 'relationship_type', mode='before')
    @classmethod
    def validate_relationship_fields(cls, v, info):
        """Validate that either related_item_id or relationship_type is provided, but not both"""
        if info.field_name == 'related_item_id':
            return v
        elif info.field_name == 'relationship_type':
            return v
        
        # This validation will be handled in the model validation
        return v

    @model_validator(mode='after')
    def validate_relationship_logic(self) -> 'RelationshipItem':
        """Ensure either related_item_id or relationship_type is provided"""
        if not self.related_item_id and not self.relationship_type:
            raise ValueError("Either related_item_id or relationship_type must be provided")
        
        if self.related_item_id and self.relationship_type:
            raise ValueError("Cannot provide both related_item_id and relationship_type")
        
        return self

    def get_related_item_id(self) -> str:
        """Get the actual related_item_id to use in processing"""
        if self.relationship_type:
            # Ensure we have a proper RelationshipType enum
            if isinstance(self.relationship_type, RelationshipType):
                return self.relationship_type.value
            else:
                # If it's not a RelationshipType, try to convert it to string
                return str(self.relationship_type)
        return self.related_item_id

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )




class GraphOverrideNode(BaseModel):
    """Developer-specified node for graph override.

    IMPORTANT:
    - 'id' is REQUIRED (relationships reference nodes by these IDs)
    - 'label' must match a node type from your registered UserGraphSchema
    - 'properties' must include ALL required fields from your schema definition

    📋 Schema Management:
    - Register schemas: POST /v1/schemas
    - View your schemas: GET /v1/schemas
    """
    id: str = Field(
        ...,
        min_length=1,
        description="**REQUIRED**: Unique identifier for this node. Must be unique within this request. Relationships reference this via source_node_id/target_node_id. Example: 'person_john_123', 'finding_cve_2024_1234'"
    )
    label: str = Field(
        ...,
        min_length=1,
        description="**REQUIRED**: Node type from your UserGraphSchema. View available types at GET /v1/schemas. System types: Memory, Person, Company, Project, Task, Insight, Meeting, Opportunity, Code"
    )
    properties: Dict[str, Any] = Field(
        ...,
        description="**REQUIRED**: Node properties matching your UserGraphSchema definition. Must include: (1) All required properties from your schema, (2) unique_identifiers if defined (e.g., 'email' for Person) to enable MERGE deduplication. View schema requirements at GET /v1/schemas"
    )

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "x-schema-ref": "GET /v1/schemas"
        }
    )

class GraphOverrideRelationship(BaseModel):
    """Developer-specified relationship for graph override.

    IMPORTANT:
    - source_node_id MUST exactly match a node 'id' from the 'nodes' array
    - target_node_id MUST exactly match a node 'id' from the 'nodes' array
    - relationship_type MUST exist in your registered UserGraphSchema
    """
    source_node_id: str = Field(
        ...,
        min_length=1,
        description="**REQUIRED**: Must exactly match the 'id' field of a node defined in the 'nodes' array of this request"
    )
    target_node_id: str = Field(
        ...,
        min_length=1,
        description="**REQUIRED**: Must exactly match the 'id' field of a node defined in the 'nodes' array of this request"
    )
    relationship_type: str = Field(
        ...,
        min_length=1,
        description="**REQUIRED**: Relationship type from your UserGraphSchema. View available types at GET /v1/schemas. System types: WORKS_FOR, WORKS_ON, HAS_PARTICIPANT, DISCUSSES, MENTIONS, RELATES_TO, CREATED_BY"
    )
    properties: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional relationship properties (e.g., {'since': '2024-01-01', 'role': 'manager'})"
    )

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "x-schema-ref": "GET /v1/schemas"
        }
    )

class GraphOverrideSpecification(BaseModel):
    """Complete graph override specification for bypassing automatic AI extraction.

    Provides full control over knowledge graph structure. All nodes and relationships
    must comply with your registered UserGraphSchema definitions.

    📋 Schema Management:
    1. Register: POST /v1/schemas (define node types, properties, unique_identifiers, relationships)
    2. View: GET /v1/schemas (see your registered schemas and available types)
    3. Use graph_override: Reference your schema types in nodes and relationships

    🔒 Validation:
    - Node IDs validated at request time (client-side via Pydantic)
    - Schema compliance validated server-side when memory is added
    - Relationships validated to ensure source/target node IDs exist
    """
    nodes: List[GraphOverrideNode] = Field(
        ...,
        min_length=1,
        description="List of nodes to create. Each node.id must be unique within this request."
    )
    relationships: List[GraphOverrideRelationship] = Field(
        default_factory=list,
        description="List of relationships. All source_node_id and target_node_id must reference node IDs defined in the 'nodes' array."
    )

    @model_validator(mode='after')
    def validate_relationship_node_ids(self):
        """Validate that all relationship node IDs reference defined nodes.

        This catches ID mismatches before the API call, providing immediate
        feedback in the developer's local environment.
        """
        node_ids = {node.id for node in self.nodes}

        for i, rel in enumerate(self.relationships):
            if rel.source_node_id not in node_ids:
                raise ValueError(
                    f"Relationship[{i}].source_node_id '{rel.source_node_id}' does not match any node ID. "
                    f"Available node IDs: {sorted(node_ids)}"
                )
            if rel.target_node_id not in node_ids:
                raise ValueError(
                    f"Relationship[{i}].target_node_id '{rel.target_node_id}' does not match any node ID. "
                    f"Available node IDs: {sorted(node_ids)}"
                )

        return self

    @model_validator(mode='after')
    def validate_unique_node_ids(self):
        """Ensure all node IDs are unique within this request.

        Duplicate IDs would cause ambiguity in relationship references.
        """
        node_ids = [node.id for node in self.nodes]
        duplicates = {nid for nid in node_ids if node_ids.count(nid) > 1}

        if duplicates:
            raise ValueError(
                f"Duplicate node IDs found: {sorted(duplicates)}. "
                f"Each node must have a unique 'id' within the request."
            )

        return self

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "x-schema-ref": "GET /v1/schemas"
        }
    )


# Property Override Models - Now imported from shared_types to avoid circular import

# Graph Generation Configuration Models

class GraphGenerationMode(str, Enum):
    """Graph generation modes"""
    AUTO = "auto"
    MANUAL = "manual"


class AutoGraphGeneration(BaseModel):
    """AI-powered graph generation with optional guidance"""
    schema_id: Optional[str] = Field(
        None,
        description="Force AI to use this specific schema instead of auto-selecting"
    )
    property_overrides: Optional[List[PropertyOverrideRule]] = Field(
        None,
        description="Override specific property values in AI-generated nodes with match conditions"
    )
    # Future extensions: enum_constraints, semantic_matching, etc.


class ManualGraphGeneration(BaseModel):
    """Complete manual control over graph structure"""
    nodes: List[GraphOverrideNode] = Field(
        ...,
        description="Exact nodes to create"
    )
    relationships: List[GraphOverrideRelationship] = Field(
        default_factory=list,
        description="Exact relationships to create"
    )
    # Future extensions: validation_mode, merge_strategy, etc.


class GraphGeneration(BaseModel):
    """Graph generation configuration"""
    mode: GraphGenerationMode = Field(
        default=GraphGenerationMode.AUTO,
        description="Graph generation mode: 'auto' (AI-powered) or 'manual' (exact specification)"
    )
    
    # Mode-specific configurations (exactly one will be used)
    auto: Optional[AutoGraphGeneration] = Field(
        None,
        description="Configuration for AI-powered graph generation"
    )
    manual: Optional[ManualGraphGeneration] = Field(
        None,
        description="Configuration for manual graph specification"
    )
    
    @model_validator(mode='after')
    def validate_mode_config(self):
        if self.mode == GraphGenerationMode.AUTO:
            if self.manual is not None:
                raise ValueError("Cannot specify 'manual' config when mode is 'auto'")
            # auto config is optional (defaults work fine)
        elif self.mode == GraphGenerationMode.MANUAL:
            if self.manual is None:
                raise ValueError("Must specify 'manual' config when mode is 'manual'")
            if self.auto is not None:
                raise ValueError("Cannot specify 'auto' config when mode is 'manual'")
        return self


class SchemaSpecificationMixin(BaseModel):
    """
    Mixin for consistent memory processing policy across all memory request types.

    Provides a unified way to control:
    1. **Graph Generation**: How knowledge graph nodes are created
       - auto: LLM extracts entities (default). If node_constraints provided, they are applied.
       - manual: You provide exact nodes (no LLM extraction)

    2. **OMO Safety Standards**: Consent, risk, and access control
       - consent: explicit, implicit, terms, none
       - risk: none, sensitive, flagged
       - acl: Read/write permissions

    3. **Schema Integration**: Reference schema-level defaults
       - schema_id: Inherit memory_policy from schema

    **Precedence**: Request-level > Schema-level > System defaults

    Note: 'structured' is deprecated alias for 'manual'. 'hybrid' is deprecated alias for 'auto'.
    """

    # PRIMARY: Unified memory policy (RECOMMENDED)
    memory_policy: Optional[MemoryPolicy] = Field(
        default=None,
        description="Unified policy for graph generation and OMO safety. "
                   "Use mode='auto' (LLM extraction, constraints applied if provided) or 'manual' (exact nodes). "
                   "Includes consent, risk, and ACL settings. "
                   "If schema_id is set, schema's memory_policy is used as defaults. "

    )

    # SHORTHAND: link_to DSL for quick node/edge constraints
    # Type alias: Union[str, List[str], Dict[str, Any]]
    link_to: Optional[Union[str, List[str], Dict[str, Any]]] = Field(
        default=None,
        description="Shorthand DSL for node/edge constraints. "
                   "Expands to memory_policy.node_constraints and edge_constraints. "
                   "Formats: "
                   "- String: 'Task:title' (semantic match on Task.title) "
                   "- List: ['Task:title', 'Person:email'] (multiple constraints) "
                   "- Dict: {'Task:title': {'set': {...}}} (with options) "
                   "Syntax: "
                   "- Node: 'Type:property', 'Type:prop=value' (exact), 'Type:prop~value' (semantic) "
                   "- Edge: 'Source->EDGE->Target:property' (arrow syntax) "
                   "- Via: 'Type.via(EDGE->Target:prop)' (relationship traversal) "
                   "- Special: '$this', '$previous', '$context:N' "
                   "Example: 'SecurityBehavior->MITIGATES->TacticDef:name' with {'create': 'never'}"
    )

    # DEPRECATED: Legacy graph generation (kept for backwards compatibility)
    graph_generation: Optional[GraphGeneration] = Field(
        default=None,
        description="DEPRECATED: Use 'memory_policy' instead. Legacy graph generation configuration. "
                   "If both memory_policy and graph_generation are provided, memory_policy takes precedence.",
        json_schema_extra={"deprecated": True}
    )

    model_config = ConfigDict(extra='allow')


class AddMemoryRequest(SchemaSpecificationMixin):
    """Request model for adding a new memory"""
    content: str = Field(
        ...,  # Makes it required
        description="The content of the memory item you want to add to memory"
    )
    type: MemoryType = Field(
        default=MemoryType.TEXT,
        description="Memory item type; defaults to 'text' if omitted"
    )

    # Organization and namespace IDs for multi-tenant scoping
    organization_id: Optional[str] = Field(
        default=None,
        description="Optional organization ID for multi-tenant memory scoping. When provided, memory is associated with this organization."
    )
    namespace_id: Optional[str] = Field(
        default=None,
        description="Optional namespace ID for multi-tenant memory scoping. When provided, memory is associated with this namespace."
    )

    # User identification (simplified)
    external_user_id: Optional[str] = Field(
        default=None,
        description="Your application's user identifier. This is the primary way to identify users. "
                   "Use this for your app's user IDs (e.g., 'user_alice_123', UUID, email). "
                   "Papr will automatically resolve or create internal users as needed."
    )
    # Deprecated fields (kept for backwards compatibility)
    user_id: Optional[str] = Field(
        default=None,
        description="DEPRECATED: Use 'external_user_id' instead. Internal Papr Parse user ID. "
                   "Most developers should not use this field directly.",
        json_schema_extra={"deprecated": True}
    )

    metadata: Optional[MemoryMetadata] = Field(
        None,
        description="Metadata used in graph and vector store for a memory item. Include role and category here."
    )
    context: Optional[List[ContextItem]] = Field(
        default_factory=list,
        description="Conversation history context for this memory. "
                   "Use for providing message history when adding a memory. "
                   "Format: [{role: 'user'|'assistant', content: '...'}]"
    )

    # =========================================================================
    # DEPRECATED: relationships_json - Use memory_policy instead
    # =========================================================================
    relationships_json: Optional[List[RelationshipItem]] = Field(
        default_factory=list,
        deprecated=True,
        description="DEPRECATED: Use 'memory_policy' instead. "
                   "Migration options: "
                   "1. Specific memory: relationships=[{source: '$this', target: 'mem_123', type: 'FOLLOWS'}] "
                   "2. Previous memory: link_to_previous_memory=True "
                   "3. Related memories: link_to_related_memories=3",
        json_schema_extra={"deprecated": True}
    )


    # memory_policy and graph_generation inherited from SchemaSpecificationMixin
    # DO NOT redefine here - use the inherited field



    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, v):
        if v in (None, ""):
            return MemoryType.TEXT
        if isinstance(v, MemoryType):
            return v
        if v == "DocumentMemoryItem":
            return MemoryType.DOCUMENT
        return v
    

    @model_validator(mode="after")
    def handle_user_id_deprecation(self):
        """Handle user_id deprecation and log warning if used."""
        if self.user_id is not None:
            # Log deprecation warning
            logger.warning(
                f"DEPRECATION WARNING: 'user_id' field is deprecated in AddMemoryRequest. "
                f"Use 'external_user_id' instead. Provided user_id: {self.user_id[:20]}..."
            )
            # If external_user_id is not set but user_id is, copy it over for backwards compat
            # Note: This assumes user_id was being used incorrectly as external_user_id
            # The validation layer will catch actual Parse IDs vs external IDs
            if self.external_user_id is None:
                # Don't auto-copy - let the user fix their code
                pass
        return self

    def get_effective_external_user_id(self) -> Optional[str]:
        """Get the effective external user ID with precedence rules.

        Precedence (highest to lowest):
        1. external_user_id at request level
        2. external_user_id in metadata
        3. None (developer is the end user)
        """
        if self.external_user_id:
            return self.external_user_id
        if self.metadata and self.metadata.external_user_id:
            return self.metadata.external_user_id
        return None

    def as_handler_dict(self) -> dict:
        """Return a dict suitable for the common_add_memory_handler, handling nested serialization."""
        handler_dict = {
            "content": self.content,
            "type": self.type,
            "metadata": self.metadata.model_dump() if self.metadata else {},
            "context": [c.model_dump() for c in self.context] if self.context else [],
            "relationships_json": [r.model_dump() for r in self.relationships_json] if self.relationships_json else [],
            "graph_generation": self.graph_generation.model_dump() if self.graph_generation else None
        }

        # Add user identification fields
        if self.external_user_id is not None:
            handler_dict["external_user_id"] = self.external_user_id
        if self.user_id is not None:
            handler_dict["user_id"] = self.user_id

        # Add multi-tenant fields if present
        if self.organization_id is not None:
            handler_dict["organization_id"] = self.organization_id
        if self.namespace_id is not None:
            handler_dict["namespace_id"] = self.namespace_id

        return handler_dict

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "Meeting with John Smith from Acme Corp about the Q4 project timeline",
                "type": "text",
                "metadata": {
                    "topics": ["product", "planning"],
                    "hierarchical_structures": "Business/Meetings/Project Planning",
                    "createdAt": "2024-10-04T10:00:00Z",
                    "location": "Conference Room A",
                    "emoji_tags": "📅,👥,📋",
                    "emotion_tags": "focused, productive",
                    "conversationId": "conv-123",
                    "sourceUrl": "https://calendar.example.com/meeting/123",
                    "external_user_id": "external_user_123",
                    "external_user_read_access": ["external_user_123", "external_user_789"],
                    "external_user_write_access": ["external_user_123"]
                },
                "context": [
                    {
                        "role": "user",
                        "content": "Let's discuss the Q4 project timeline with John"
                    },
                    {
                        "role": "assistant", 
                        "content": "I'll help you prepare for the timeline discussion. What are your key milestones?"
                    }
                ],
                # OPTION 1: Automatic graph creation (most common - just omit graph_override)
                # The system will automatically extract entities and relationships using AI
                
                # OPTION 2: Override with your own graph structure (advanced use case)
                "graph_override": {
                    "nodes": [
                        {
                            "id": "person_john_smith",
                            "label": "Person",
                            "properties": {
                                "name": "John Smith",
                                "role": "Project Manager", 
                                "description": "Senior PM at Acme Corp"
                            }
                        },
                        {
                            "id": "company_acme_corp",
                            "label": "Company",
                            "properties": {
                                "name": "Acme Corp",
                                "description": "Client company for Q4 project"
                            }
                        }
                    ],
                    "relationships": [
                        {
                            "source_node_id": "person_john_smith",
                            "target_node_id": "company_acme_corp", 
                            "relationship_type": "WORKS_FOR",
                            "properties": {
                                "role": "Project Manager"
                            }
                        }
                    ]
                }
            }
        },
        populate_by_name=True,
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

class UpdateMemoryRequest(SchemaSpecificationMixin):
    """Request model for updating an existing memory.

    Inherits memory_policy from SchemaSpecificationMixin for controlling:
    - Graph generation mode (auto, manual)
    - Node constraints for LLM extraction (applied in auto mode when present)
    - OMO safety standards (consent, risk, ACL)
    """
    content: Optional[str] = Field(
        None,
        description="The new content of the memory item"
    )
    type: Optional[MemoryType] = Field(
        None,
        description="Content type of the memory item"
    )
    metadata: Optional[MemoryMetadata] = Field(
        None,
        description="Updated metadata for Neo4J and Pinecone"
    )
    context: Optional[List[ContextItem]] = Field(
        None,
        description="Updated context for the memory item"
    )
    relationships_json: Optional[List[RelationshipItem]] = Field(
        None,
        description="Updated relationships for Graph DB (neo4J)"
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Optional organization ID for multi-tenant memory scoping. When provided, update is scoped to memories within this organization."
    )
    namespace_id: Optional[str] = Field(
        default=None,
        description="Optional namespace ID for multi-tenant memory scoping. When provided, update is scoped to memories within this namespace."
    )
    # memory_policy and graph_generation inherited from SchemaSpecificationMixin

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "Updated meeting notes from the product planning session",
                "type": "text",
                "metadata": {
                    "topics": "product, planning, updates",
                    "hierarchical structures": "Business/Planning/Product/Updates",
                    "emoji tags": "📊,💡,📝,✨",
                    "emotion tags": "focused, productive, satisfied"
                },
                "context": [
                    {
                        "role": "user",
                        "content": "Let's update the Q2 product roadmap"
                    },
                    {
                        "role": "assistant",
                        "content": "I'll help you update the roadmap. What changes would you like to make?"
                    }
                ],
                "relationships_json": [
                    {
                        "related_item_id": "previous_memory_item_id",
                        "relation_type": "updates",
                        "related_item_type": "TextMemoryItem",
                        "metadata": {
                            "relevance": "high"
                        }
                    }
                ]
            }
        },
        populate_by_name=True,
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

class BatchMemoryRequest(SchemaSpecificationMixin):
    """Request model for batch adding memories"""
    # Primary user identification
    external_user_id: Optional[str] = Field(
        default=None,
        description="Your application's user identifier for all memories in the batch. This is the primary way to identify users. "
                   "Papr will automatically resolve or create internal users as needed."
    )
    # Deprecated field (kept for backwards compatibility)
    user_id: Optional[str] = Field(
        default=None,
        description="DEPRECATED: Use 'external_user_id' instead. Internal Papr Parse user ID.",
        json_schema_extra={"deprecated": True}
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Optional organization ID for multi-tenant batch memory scoping. When provided, all memories in the batch are associated with this organization."
    )
    namespace_id: Optional[str] = Field(
        default=None,
        description="Optional namespace ID for multi-tenant batch memory scoping. When provided, all memories in the batch are associated with this namespace."
    )
    # schema_id, graph_override inherited from SchemaSpecificationMixin
    memories: List[AddMemoryRequest] = Field(
        ...,  # Makes it required
        description="List of memory items to add in batch",
        min_length=1,  # Require at least one memory
        max_length=50  # Limit batch size to prevent overload
    )
    batch_size: Optional[int] = Field(
        default=10,
        description="Number of items to process in parallel",
        ge=1,
        le=50
    )
    webhook_url: Optional[str] = Field(
        default=None,
        description="Optional webhook URL to notify when batch processing is complete. The webhook will receive a POST request with batch completion details."
    )
    webhook_secret: Optional[str] = Field(
        default=None,
        description="Optional secret key for webhook authentication. If provided, will be included in the webhook request headers as 'X-Webhook-Secret'."
    )

    @field_validator('memories')
    @classmethod
    def validate_batch_sizes(cls, v: List[AddMemoryRequest]) -> List[AddMemoryRequest]:
        """Validate content sizes for the batch"""
        MAX_CONTENT_LENGTH = int(env.get('MAX_CONTENT_LENGTH', 15000))
        MAX_TOTAL_BATCH_CONTENT = int(env.get('MAX_TOTAL_BATCH_CONTENT', 750000))
        MAX_TOTAL_BATCH_STORAGE = int(env.get('MAX_TOTAL_BATCH_STORAGE', 750000))
        
        total_content_size = 0
        total_storage_size = 0
        
        for memory in v:
            content_size = len(memory.content.encode('utf-8'))
            total_content_size += content_size
            total_storage_size += content_size
            
            if content_size > MAX_CONTENT_LENGTH:
                raise ValueError(
                    f"Individual content size ({content_size} bytes) exceeds maximum limit of {MAX_CONTENT_LENGTH} bytes"
                )
        
        if total_content_size > MAX_TOTAL_BATCH_CONTENT:
            raise ValueError(
                f"Total batch content size ({total_content_size} bytes) exceeds maximum limit of {MAX_TOTAL_BATCH_CONTENT} bytes"
            )
            
        if total_storage_size > MAX_TOTAL_BATCH_STORAGE:
            raise ValueError(
                f"Total batch storage size ({total_storage_size} bytes) exceeds maximum limit of {MAX_TOTAL_BATCH_STORAGE} bytes"
            )
            
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "internal_user_id_12345",
                "external_user_id": "external_user_abcde",
                "memories": [
                    {
                        "content": "Meeting notes from the product planning session",
                        "type": "text",
                        "metadata": {
                            "topics": "product, planning",
                            "hierarchical structures": "Business/Planning/Product",
                            "createdAt": "2024-03-21T10:00:00Z",
                            "emoji tags": "📊,💡,📝",
                            "emotion tags": "focused, productive"
                        }
                    },
                    {
                        "content": "Follow-up tasks from the planning meeting",
                        "type": "text",
                        "metadata": {
                            "topics": "tasks, planning",
                            "hierarchical structures": "Business/Tasks/Planning",
                            "createdAt": "2024-03-21T11:00:00Z",
                            "emoji tags": "✅,📋",
                            "emotion tags": "organized"
                        }
                    }
                ],
                "batch_size": 10
            }
        }
    )

class OptimizedAuthResponse(BaseModel):
    """Response model for optimized authentication with multi-tenant support"""
    developer_id: str = Field(..., description="The authenticated developer's user ID")
    end_user_id: str = Field(..., description="The resolved end user ID for the operation")
    session_token: Optional[str] = Field(None, description="Session token if available")
    user_info: Optional[Dict[str, Any]] = Field(None, description="Additional user information")
    api_key: Optional[str] = Field(None, description="API key if used for authentication")
    workspace_id: Optional[str] = Field(None, description="Selected workspace ID")
    is_qwen_route: Optional[bool] = Field(None, description="Whether this is a Qwen route")
    user_roles: List[str] = Field(default_factory=list, description="User roles")
    user_workspace_ids: List[str] = Field(default_factory=list, description="User workspace IDs")
    updated_metadata: Optional[MemoryMetadata] = Field(None, description="Updated metadata from user resolution")
    updated_batch_request: Optional[BatchMemoryRequest] = Field(None, description="Updated batch request with resolved user IDs and ACLs")

    # Multi-tenant fields
    organization_id: Optional[str] = Field(None, description="Organization ID for multi-tenant support")
    namespace_id: Optional[str] = Field(None, description="Namespace ID for multi-tenant support")
    is_legacy_auth: bool = Field(True, description="Whether this is legacy authentication (True) or organization-based (False)")
    auth_type: Literal["legacy", "organization"] = Field("legacy", description="Type of authentication used")
    api_key_info: Optional[Dict[str, Any]] = Field(None, description="Additional API key metadata for organization-based auth")
    user_schemas: Optional[List] = Field(default_factory=list, description="User's active schemas fetched in parallel for search optimization")
    cached_schema: Optional[Dict[str, Any]] = Field(None, description="Cached schema patterns from Parse for agentic search optimization")
    
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='allow'
    )

    @model_validator(mode='after')
    def validate_multi_tenant_fields(self):
        """Validate that organization-based auth has required fields"""
        if self.auth_type == "organization":
            if not self.organization_id or not self.namespace_id:
                raise ValueError("Organization and namespace IDs are required for organization-based authentication")
        return self

class SearchResponse(BaseModel):
    code: int = Field(default=200, description="HTTP status code")
    status: str = Field(default="success", description="'success' or 'error'")
    data: Optional[SearchResult] = Field(default=None, description="Search results if successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    details: Optional[Any] = Field(default=None, description="Additional error details or context")
    search_id: Optional[str] = Field(default=None, description="Unique identifier for this search query, maps to QueryLog objectId in Parse Server")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": 200,
                "status": "success",
                "data": {
                    "memories": [],
                    "nodes": []
                },
                "error": None,
                "details": None,
                "search_id": "abc123def456"
            }
        }
    )

    @classmethod
    def success(cls, data: SearchResult, code: int = 200, search_id: Optional[str] = None):
        return cls(
            code=code,
            status="success",
            data=data,
            error=None,
            details=None,
            search_id=search_id
        )

    @classmethod
    def failure(cls, error: str, code: int = 400, details: Any = None, search_id: Optional[str] = None):
        return cls(
            code=code,
            status="error",
            data=None,
            error=error,
            details=details,    
            search_id=search_id
        )


class EmbeddingFormat(str, Enum):
    """Embedding format options for sync tiers"""
    INT8 = "int8"  # Quantized to INT8, 4x smaller, efficient for network transfer
    FLOAT32 = "float32"  # Full precision float32, recommended for CoreML/ANE fp16 models


class SyncTiersRequest(BaseModel):
    """Request model for sync tiers endpoint"""
    max_tier0: int = Field(
        default=300, 
        ge=0, 
        le=2000, 
        description="Max Tier 0 items (goals/OKRs/use-cases)"
    )
    max_tier1: int = Field(
        default=1000, 
        ge=0, 
        le=5000, 
        description="Max Tier 1 items (hot memories)"
    )
    workspace_id: Optional[str] = Field(
        default=None, 
        description="Optional workspace id to scope tiers"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional internal user ID to filter tiers by a specific user. If not provided, uses authenticated user."
    )
    external_user_id: Optional[str] = Field(
        default=None,
        description="Optional external user ID to filter tiers by a specific external user. If both user_id and external_user_id are provided, user_id takes precedence."
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Optional organization ID for multi-tenant scoping. When provided, tiers are scoped to memories within this organization."
    )
    namespace_id: Optional[str] = Field(
        default=None,
        description="Optional namespace ID for multi-tenant scoping. When provided, tiers are scoped to memories within this namespace."
    )
    include_embeddings: bool = Field(
        default=False, 
        description="Include embeddings in the response. Format controlled by embedding_format parameter."
    )
    embedding_format: EmbeddingFormat = Field(
        default=EmbeddingFormat.INT8,
        description="Embedding format: 'int8' (quantized, 4x smaller, default for efficiency), 'float32' (full precision, recommended for CoreML/ANE fp16 models). Only applies to Tier1; Tier0 always uses float32 when embeddings are included."
    )
    embed_model: str = Field(
        default="Qwen4B", 
        description="Embedding model hint: 'sbert' or 'bigbird' or 'Qwen4B'"
    )
    embed_limit: int = Field(
        default=200, 
        ge=0, 
        le=1000, 
        description="Max items to embed per tier to control latency"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional internal user ID to filter tiers by a specific user. If not provided, results are not filtered by user. If both user_id and external_user_id are provided, user_id takes precedence."
    )
    external_user_id: Optional[str] = Field(
        default=None,
        description="Optional external user ID to filter tiers by a specific external user. If both user_id and external_user_id are provided, user_id takes precedence."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "max_tier0": 300,
                "max_tier1": 1000,
                "workspace_id": "workspace_123",
                "include_embeddings": False,
                "embed_model": "sbert",
                "embed_limit": 200,
                "user_id": "internal_user_123",
                "external_user_id": "external_user_abc"
            }
        }
    )


class SyncTiersResponse(BaseModel):
    """Response model for sync tiers endpoint"""
    code: int = Field(default=200, description="HTTP status code")
    status: str = Field(default="success", description="'success' or 'error'")
    tier0: List[Memory] = Field(
        default_factory=list, 
        description="Tier 0 items (goals/OKRs/use-cases)"
    )
    tier1: List[Memory] = Field(
        default_factory=list, 
        description="Tier 1 items (hot memories)"
    )
    transitions: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Transition items between tiers"
    )
    next_cursor: Optional[str] = Field(
        default=None, 
        description="Cursor for pagination"
    )
    has_more: bool = Field(
        default=False, 
        description="Whether there are more items available"
    )
    error: Optional[str] = Field(
        default=None, 
        description="Error message if failed"
    )
    details: Optional[Any] = Field(
        default=None, 
        description="Additional error details or context"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": 200,
                "status": "success",
                "tier0": [
                    {
                        "id": "goal_123",
                        "content": "Improve API performance",
                        "type": "goal",
                        "topics": ["performance", "api"],
                        "metadata": {"sourceType": "papr", "class": "goal"}
                    }
                ],
                "tier1": [
                    {
                        "id": "memory_456",
                        "content": "Customer complained about slow API response times",
                        "type": "text",
                        "topics": ["customer", "api", "performance"],
                        "metadata": {"sourceType": "papr"}
                    }
                ],
                "transitions": [],
                "next_cursor": None,
                "has_more": False,
                "error": None,
                "details": None
            }
        }
    )

    @classmethod
    def success(cls, tier0: List[Memory], tier1: List[Memory], **kwargs):
        return cls(
            code=200,
            status="success",
            tier0=tier0,
            tier1=tier1,
            transitions=kwargs.get("transitions", []),
            next_cursor=kwargs.get("next_cursor"),
            has_more=kwargs.get("has_more", False),
            error=None,
            details=None
        )

    @classmethod
    def failure(cls, error: str, code: int = 500, details: Any = None):
        return cls(
            code=code,
            status="error",
            tier0=[],
            tier1=[],
            transitions=[],
            next_cursor=None,
            has_more=False,
            error=error,
            details=details
        )

class NodeConverter:
    @staticmethod
    def normalize_datetime_keys(d: dict) -> dict:
        """Ensure only camelCase datetime keys exist, remove snake_case ones."""
        d = d.copy()
        # Prefer camelCase if both exist, otherwise use snake_case
        if 'createdAt' not in d and 'created_at' in d:
            d['createdAt'] = d['created_at']
        if 'updatedAt' not in d and 'updated_at' in d:
            d['updatedAt'] = d['updated_at']
        # Remove snake_case keys if present
        d.pop('created_at', None)
        d.pop('updated_at', None)
        return d

    @staticmethod
    def _parse_access_lists(node_dict: dict) -> dict:
        """Convert string representations of lists to actual lists"""
        access_fields = [
            'user_read_access',
            'user_write_access',
            'workspace_read_access',
            'workspace_write_access',
            'role_read_access',
            'role_write_access'
        ]
        
        parsed_dict = node_dict.copy()
        for field in access_fields:
            if field in parsed_dict:
                try:
                    # Handle string representation of list
                    if isinstance(parsed_dict[field], str):
                        # Remove quotes and brackets, split by comma
                        value = parsed_dict[field].strip('[]\'\"')
                        if value:
                            # Split by comma and clean up each item
                            parsed_dict[field] = [item.strip().strip('\'\"') for item in value.split(',')]
                        else:
                            parsed_dict[field] = []
                except Exception as e:
                    logger.warning(f"Error parsing {field}: {str(e)}")
                    parsed_dict[field] = []
                    
        return parsed_dict

    @staticmethod
    def _parse_string_to_list(value: Union[str, List[str], None]) -> List[str]:
        """Convert string or list to proper list format"""
        if not value:
            return []
        if isinstance(value, list):
            return value
        # Must be string at this point since we've defined Union[str, List[str], None]
        return [item.strip() for item in value.split(',')]
    
    @staticmethod
    def get_id(parsed_dict, *fallback_keys):
        for key in ('id',) + fallback_keys:
            if key in parsed_dict and parsed_dict[key]:
                return parsed_dict[key]
        # Try to find a key that looks like an id (e.g., <id>)
        for k, v in parsed_dict.items():
            if k.lower() == 'id' and v:
                return v
        # As a last resort, generate a UUID
        return str(uuid.uuid4())
    
    @staticmethod
    def convert_to_neo_node(node_dict: dict, primary_label: str) -> Optional[NeoNode]:
        """
        Convert a dictionary of node properties to the appropriate NeoNode object
        
        Args:
            node_dict (dict): Dictionary containing node properties
            primary_label (str): Primary label of the node ('Task', 'Memory', etc.)
            
        Returns:
            Optional[NeoNode]: Converted node or None if conversion fails
        """
        logger.info(f"[LOG] Entered convert_to_neo_node with primary_label={primary_label}")
        logger.info(f"[LOG] node_dict at entry: {json.dumps(node_dict, default=str, indent=2)}")
        try:
            # Normalize datetime keys before any further processing
            node_dict = NodeConverter.normalize_datetime_keys(node_dict)
            logger.info(f"[DEBUG] convert_to_neo_node: primary_label={primary_label}, node_dict={node_dict}")
            # Parse access control lists first
            logger.info(f"[DEBUG] Before _parse_access_lists: node_dict={node_dict}")
            parsed_dict = NodeConverter._parse_access_lists(node_dict)
            logger.info(f"[DEBUG] After _parse_access_lists: parsed_dict={parsed_dict}")

            # Handle topics and other list fields
            if 'topics' in parsed_dict:
                logger.info(f"[DEBUG] Before _parse_string_to_list for topics: {parsed_dict['topics']}")
                parsed_dict['topics'] = NodeConverter._parse_string_to_list(parsed_dict['topics'])
                logger.info(f"[DEBUG] After _parse_string_to_list for topics: {parsed_dict['topics']}")
            if 'emotion_tags' in parsed_dict:
                logger.info(f"[DEBUG] Before _parse_string_to_list for emotion_tags: {parsed_dict.get('emotionTags', parsed_dict.get('emotion_tags', ''))}")
                parsed_dict['emotion_tags'] = NodeConverter._parse_string_to_list(parsed_dict.get('emotionTags', parsed_dict.get('emotion_tags', '')))
                logger.info(f"[DEBUG] After _parse_string_to_list for emotion_tags: {parsed_dict['emotion_tags']}")
            if 'emoji_tags' in parsed_dict:
                logger.info(f"[DEBUG] Before _parse_string_to_list for emoji_tags: {parsed_dict.get('emojiTags', parsed_dict.get('emoji_tags', ''))}")
                parsed_dict['emoji_tags'] = NodeConverter._parse_string_to_list(parsed_dict.get('emojiTags', parsed_dict.get('emoji_tags', '')))
                logger.info(f"[DEBUG] After _parse_string_to_list for emoji_tags: {parsed_dict['emoji_tags']}")
                
            # Rename hierarchicalStructures to hierarchical_structures
            if 'hierarchicalStructures' in parsed_dict:
                logger.info(f"[DEBUG] Renaming hierarchicalStructures to hierarchical_structures: {parsed_dict['hierarchicalStructures']}")
                parsed_dict['hierarchical_structures'] = parsed_dict.pop('hierarchicalStructures')

            logger.info(f"primary_label: {primary_label}")
            
            # UNIFIED APPROACH: Treat all nodes (system and custom) the same way
            # No validation - just return the data as-is using FlexibleCustomNode
            # This allows developers to customize system nodes with their own schema
            
            # Get base properties for any node type
            base_fields = NodeConverter._get_base_properties(parsed_dict)
            
            # Add the node ID
            base_fields['id'] = NodeConverter.get_id(parsed_dict)
            
            # Preserve ALL original properties, not just base ones
            # This ensures custom fields are preserved and system nodes can be customized
            all_fields = base_fields.copy()
            for key, value in parsed_dict.items():
                if key not in all_fields and value is not None:
                    all_fields[key] = value
            
            logger.info(f"Node fields for {primary_label}: {list(all_fields.keys())}")
            
            try:
                # Create a dynamic NodeLabel that preserves the original type
                node_label = NodeLabel(primary_label)
                
                # Use the flexible property container that accepts any fields
                # No validation - just pass through all properties
                props = FlexibleCustomNode(**all_fields)
                
                logger.info(f"Created NeoNode with label '{primary_label}': {node_label}")
                return NeoNode(label=node_label, properties=props)
                
            except Exception as e:
                logger.error(f"Failed to create node for {primary_label}: {e}")
                logger.error(f"All fields: {all_fields}")
                
                # Fallback: try with minimal fields
                try:
                    # Create minimal fields ensuring id is present
                    minimal_fields = {
                        'id': all_fields.get('id', ''),
                    }
                    # Add any other fields that are not None
                    for key, value in all_fields.items():
                        if value is not None:
                            minimal_fields[key] = value
                            
                    props = FlexibleCustomNode(**minimal_fields)
                    logger.info(f"Created node '{primary_label}' using fallback minimal fields")
                    return NeoNode(label=node_label, properties=props)
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback creation also failed for {primary_label}: {fallback_error}")
                    return None

        except ValidationError as e:
            logger.error(f"Validation error for {primary_label} node:")
            logger.error(f"Error details: {str(e)}")
            logger.error(f"Failed properties: {parsed_dict}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing {primary_label} node: {str(e)}")
            logger.error(f"Node data: {parsed_dict}")
            return None
        finally:
            logger.debug(f"[LOG] Exiting convert_to_neo_node for primary_label={primary_label}")

    @staticmethod
    def _get_base_properties(node_dict: dict) -> dict:
        """Extract common base properties from node dictionary, normalizing datetime keys."""
        node_dict = NodeConverter.normalize_datetime_keys(node_dict)
        return {
            'user_id': node_dict.get('user_id'),
            'workspace_id': node_dict.get('workspace_id'),
            'sourceType': node_dict.get('sourceType'),
            'sourceUrl': node_dict.get('sourceUrl'),
            'pageId': node_dict.get('pageId'),
            'conversationId': node_dict.get('conversationId'),
            'createdAt': node_dict.get('createdAt'),
            'updatedAt': node_dict.get('updatedAt'),
            'user_read_access': node_dict.get('user_read_access', []),
            'user_write_access': node_dict.get('user_write_access', []),
            'workspace_read_access': node_dict.get('workspace_read_access', []),
            'workspace_write_access': node_dict.get('workspace_write_access', []),
            'role_read_access': node_dict.get('role_read_access', []),
            'role_write_access': node_dict.get('role_write_access', []),
            'external_user_read_access': node_dict.get('external_user_read_access', []),
            'external_user_write_access': node_dict.get('external_user_write_access', []),
            'external_user_id': node_dict.get('external_user_id')
        }

