from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Dict, Any, List, Optional, Union, Literal, TYPE_CHECKING
from enum import Enum
from datetime import datetime, timezone
from uuid import uuid4
import re

if TYPE_CHECKING:
    from models.shared_types import MemoryPolicy, NodeConstraint, EdgeConstraint

class PropertyType(str, Enum):
    STRING = "string"
    INTEGER = "integer" 
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    DATETIME = "datetime"
    OBJECT = "object"

class PropertyDefinition(BaseModel):
    """Property definition for nodes/relationships"""
    type: PropertyType
    required: bool = False
    default: Optional[Any] = None
    description: Optional[str] = None

    # Validation rules (only included if explicitly set)
    min_length: Optional[int] = None  # For strings
    max_length: Optional[int] = None  # For strings
    min_value: Optional[float] = None  # For numbers
    max_value: Optional[float] = None  # For numbers
    enum_values: Optional[List[str]] = Field(None, max_length=15, description="List of allowed enum values (max 15)")  # For enumerated values
    pattern: Optional[str] = None  # Regex pattern for strings

    model_config = ConfigDict(extra='forbid', exclude_none=True)
    
    @field_validator('enum_values')
    @classmethod
    def validate_enum_values(cls, v):
        if v is not None:
            if len(v) == 0:
                raise ValueError("enum_values cannot be empty if provided")
            if len(v) > 15:
                raise ValueError("enum_values cannot contain more than 15 values")
            # Check for duplicates
            if len(v) != len(set(v)):
                raise ValueError("enum_values cannot contain duplicate values")
            # Ensure all values are non-empty strings
            for enum_val in v:
                if not isinstance(enum_val, str) or not enum_val.strip():
                    raise ValueError("All enum_values must be non-empty strings")
        return v
    
    @field_validator('default')
    @classmethod
    def validate_default_against_enum(cls, v, info):
        """Validate that default value is in enum_values if enum is specified"""
        if v is not None and 'enum_values' in info.data:
            enum_values = info.data['enum_values']
            if enum_values is not None and str(v) not in enum_values:
                raise ValueError(f"Default value '{v}' must be one of the enum_values: {enum_values}")
        return v

class UserNodeType(BaseModel):
    """
    User-defined node type with optional inline constraint.

    The `constraint` field allows defining default matching/creation behavior
    directly within the node type definition. This replaces the need to put
    constraints only in memory_policy.node_constraints.

    Schema-level constraints:
    - `node_type` is implicit (taken from parent UserNodeType.name)
    - Defines default matching strategy via `search.properties`
    - Can be overridden per-memory via memory_policy.node_constraints

    Example:
        UserNodeType(
            name="Task",
            label="Task",
            properties={
                "id": PropertyDefinition(type="string"),
                "title": PropertyDefinition(type="string", required=True)
            },
            constraint=NodeConstraint(
                search=SearchConfig(properties=[
                    PropertyMatch(name="id", mode="exact"),
                    PropertyMatch(name="title", mode="semantic", threshold=0.85)
                ]),
                create="auto"
            )
        )
    """
    name: str = Field(..., pattern=r'^[A-Za-z][A-Za-z0-9_]*$')  # Valid identifier
    label: str  # Display name
    description: Optional[str] = None
    properties: Dict[str, PropertyDefinition] = Field(
        default_factory=dict,
        description="Node properties (max 10 per node type)"
    )
    required_properties: List[str] = Field(default_factory=list)

    # Node merging/deduplication (DEPRECATED: Use constraint.search.properties instead)
    unique_identifiers: List[str] = Field(
        default_factory=list,
        description="DEPRECATED: Use 'constraint.search.properties' instead. "
                   "Properties that uniquely identify this node type. "
                   "Example: ['name', 'email'] for Customer nodes."
    )

    # Node constraint - defines default matching/creation behavior for this node type
    constraint: Optional["NodeConstraint"] = Field(
        default=None,
        description="Default constraint for this node type. Defines: "
                   "1. search.properties - unique identifiers and how to match them "
                   "2. create - 'auto' (create if not found) or 'never' (controlled vocabulary) "
                   "3. when - conditional application with logical operators "
                   "4. set - default property values. "
                   "Note: node_type is implicit (taken from this UserNodeType.name)."
    )

    # Link-only shorthand (equivalent to @link_only decorator)
    link_only: bool = Field(
        default=False,
        description="Shorthand for constraint with create='never'. "
                   "When True, only links to existing nodes (controlled vocabulary). "
                   "Equivalent to @link_only decorator. If constraint is also provided, "
                   "link_only=True will override constraint.create to 'never'."
    )

    # Visual/UI properties
    color: Optional[str] = "#3498db"  # Hex color for UI
    icon: Optional[str] = None  # Icon identifier

    model_config = ConfigDict(extra='forbid')
    
    @field_validator('properties')
    @classmethod
    def validate_properties(cls, v):
        if len(v) > 10:
            raise ValueError(f"Node type cannot have more than 10 properties (found {len(v)})")
        return v
    
    @field_validator('required_properties')
    @classmethod
    def validate_required_properties(cls, v, info):
        if 'properties' in info.data:
            properties = info.data['properties']
            for req_prop in v:
                if req_prop not in properties:
                    raise ValueError(f"Required property '{req_prop}' not found in properties")
        return v
    
    @field_validator('unique_identifiers')
    @classmethod
    def validate_unique_identifiers(cls, v, info):
        if 'properties' in info.data:
            properties = info.data['properties']
            for unique_prop in v:
                if unique_prop not in properties:
                    raise ValueError(f"Unique identifier property '{unique_prop}' not found in properties")
        return v

    def model_post_init(self, __context):
        """Apply link_only to constraint after initialization."""
        if self.link_only:
            # Import here to avoid circular dependency issues
            from models.shared_types import NodeConstraint
            if self.constraint is None:
                # Create a new constraint with create='never'
                object.__setattr__(self, 'constraint', NodeConstraint(create="never"))
            else:
                # Override the existing constraint's create field
                object.__setattr__(self.constraint, 'create', 'never')


class UserRelationshipType(BaseModel):
    """
    User-defined relationship type with optional inline constraint.

    The `constraint` field allows defining default matching/creation behavior
    directly within the relationship type definition. This mirrors the pattern
    used in UserNodeType.constraint for nodes.

    Schema-level edge constraints:
    - `edge_type` is implicit (taken from parent UserRelationshipType.name)
    - Defines default target node matching strategy via `search.properties`
    - Can be overridden per-memory via memory_policy.edge_constraints

    Example:
        UserRelationshipType(
            name="MITIGATES",
            label="Mitigates",
            allowed_source_types=["SecurityBehavior"],
            allowed_target_types=["TacticDef"],
            constraint=EdgeConstraint(
                search=SearchConfig(properties=[
                    PropertyMatch(name="name", mode="semantic", threshold=0.90)
                ]),
                create="never"  # Controlled vocabulary - only link to existing targets
            )
        )
    """
    name: str = Field(..., pattern=r'^[A-Z][A-Z0-9_]*$')  # Convention: UPPER_CASE
    label: str  # Display name
    description: Optional[str] = None
    properties: Dict[str, PropertyDefinition] = Field(default_factory=dict)

    # Relationship constraints
    allowed_source_types: List[str]  # Which node types can be source
    allowed_target_types: List[str]  # Which node types can be target
    cardinality: Literal["one-to-one", "one-to-many", "many-to-many"] = "many-to-many"

    # Edge constraint - defines default matching/creation behavior for this relationship type
    constraint: Optional["EdgeConstraint"] = Field(
        default=None,
        description="Default constraint for this relationship type. Defines: "
                   "1. search.properties - how to find existing target nodes "
                   "2. create - 'auto' (create if not found) or 'never' (controlled vocabulary) "
                   "3. when - conditional application with logical operators "
                   "4. set - default edge property values. "
                   "Note: edge_type is implicit (taken from this UserRelationshipType.name)."
    )

    # Link-only shorthand (equivalent to @link_only decorator)
    link_only: bool = Field(
        default=False,
        description="Shorthand for constraint with create='never'. "
                   "When True, only links to existing target nodes (controlled vocabulary). "
                   "Equivalent to @link_only decorator. If constraint is also provided, "
                   "link_only=True will override constraint.create to 'never'."
    )

    # Visual/UI properties
    color: Optional[str] = "#e74c3c"  # Hex color for UI

    model_config = ConfigDict(extra='forbid')

    def model_post_init(self, __context):
        """Apply link_only to constraint after initialization."""
        if self.link_only:
            # Import here to avoid circular dependency issues
            from models.shared_types import EdgeConstraint
            if self.constraint is None:
                # Create a new constraint with create='never'
                object.__setattr__(self, 'constraint', EdgeConstraint(create="never"))
            else:
                # Override the existing constraint's create field
                object.__setattr__(self.constraint, 'create', 'never')


class SchemaStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class SchemaScope(str, Enum):
    """Schema scopes available through the API"""
    PERSONAL = "personal"           # Private to user
    WORKSPACE = "workspace"         # Shared within workspace (legacy)
    NAMESPACE = "namespace"         # Shared within namespace (environment-specific)
    ORGANIZATION = "organization"   # Shared across all namespaces in organization

class UserGraphSchema(BaseModel):
    """Complete user-defined graph schema"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    version: str = Field(default="1.0.0", pattern=r'^\d+\.\d+\.\d+$')
    
    # Ownership
    user_id: Optional[Union[str, Dict[str, Any]]] = None  # Set automatically from authentication (string or Parse Pointer)
    workspace_id: Optional[Union[str, Dict[str, Any]]] = None  # String or Parse Pointer (legacy)
    
    # Multi-tenant ownership (NEW) - Parse Pointers
    organization: Optional[Union[str, Dict[str, Any]]] = None  # Organization this schema belongs to (Parse Pointer)
    namespace: Optional[Union[str, Dict[str, Any]]] = None     # Namespace this schema belongs to (Parse Pointer)
    
    # Schema definitions
    node_types: Dict[str, UserNodeType] = Field(
        default_factory=dict,
        description="Custom node types (max 10 per schema)"
    )
    relationship_types: Dict[str, UserRelationshipType] = Field(
        default_factory=dict,
        description="Custom relationship types (max 20 per schema)"
    )

    # Memory Policy - default processing rules for memories using this schema
    memory_policy: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Default memory policy for memories using this schema. "
                   "Includes mode ('auto', 'manual'), node_constraints (applied in auto mode when present), "
                   "and OMO safety settings (consent, risk). Memory-level policies override schema-level."
    )

    # Metadata
    status: SchemaStatus = SchemaStatus.DRAFT
    scope: SchemaScope = SchemaScope.ORGANIZATION  # Default to organization scope for multi-tenant isolation
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    
    # Access control
    read_access: List[str] = Field(default_factory=list)
    write_access: List[str] = Field(default_factory=list)
    
    # Usage tracking
    usage_count: int = 0
    last_used_at: Optional[datetime] = None
    
    model_config = ConfigDict(extra='forbid')
    
    @field_validator('node_types')
    @classmethod
    def validate_node_types(cls, v):
        if not v:
            raise ValueError("Schema must have at least one node type")
        if len(v) > 10:
            raise ValueError(f"Schema cannot have more than 10 node types (found {len(v)})")
        return v
    
    @field_validator('relationship_types')
    @classmethod
    def validate_relationship_types(cls, v):
        if len(v) > 20:
            raise ValueError(f"Schema cannot have more than 20 relationship types (found {len(v)})")
        return v


    @field_validator('created_at', 'updated_at', 'last_used_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        """Parse Parse Server date format {'__type': 'Date', 'iso': '...'} to datetime"""
        if v is None:
            return v
        if isinstance(v, datetime):
            return v
        if isinstance(v, dict) and v.get('__type') == 'Date':
            iso_str = v.get('iso')
            if iso_str:
                return datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

# Response models for API
class SchemaResponse(BaseModel):
    """Response model for schema operations"""
    success: bool
    data: Optional[UserGraphSchema] = None
    error: Optional[str] = None
    code: int = 200

class SchemaListResponse(BaseModel):
    """Response model for listing schemas"""
    success: bool
    data: Optional[List[UserGraphSchema]] = None
    error: Optional[str] = None
    code: int = 200
    total: int = 0


# ============================================================================
# Resolve forward references after all models are defined
# ============================================================================
# Import NodeConstraint and EdgeConstraint, then rebuild models to resolve forward references
from models.shared_types import NodeConstraint, EdgeConstraint
UserNodeType.model_rebuild()
UserRelationshipType.model_rebuild()
