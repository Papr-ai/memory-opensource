# Shared types for memory, parse_server, and structured_outputs
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import Optional, List, Dict, Any, Union, Literal, TYPE_CHECKING
from enum import Enum
import logging
from datetime import datetime, timezone, UTC
logger = logging.getLogger(__name__)
import json

if TYPE_CHECKING:
    from models.omo import OpenMemoryObject
 

# SchemaSpecificationMixin will be imported locally in UploadDocumentRequest to avoid circular imports

# Module-level registries for custom types (to avoid enum class attribute issues)
_custom_labels_registry = set()
_custom_relationships_registry = set()


class MemoryType(str, Enum):
    """Valid memory types"""
    TEXT = "text"
    CODE_SNIPPET = "code_snippet"
    DOCUMENT = "document"


class MessageRole(str, Enum):
    """Role of the message sender"""
    USER = "user"
    ASSISTANT = "assistant"


class UserMemoryCategory(str, Enum):
    """Memory categories for user messages"""
    PREFERENCE = "preference"
    TASK = "task"
    GOAL = "goal"
    FACT = "fact"  # Changed from FACTS to FACT for consistency
    CONTEXT = "context"


class AssistantMemoryCategory(str, Enum):
    """Memory categories for assistant messages"""
    SKILLS = "skills"
    LEARNING = "learning"
    TASK = "task"  # Added task for assistants
    GOAL = "goal"  # Added goal for assistants
    FACT = "fact"  # Added fact for assistants
    CONTEXT = "context"  # Added context for assistants

# Enums for node labels and relationship types
class NodeLabel(str, Enum):
    Memory = "Memory"
    Person = "Person"
    Company = "Company"
    Project = "Project"
    Task = "Task"
    Insight = "Insight"
    Meeting = "Meeting"
    Opportunity = "Opportunity"
    Code = "Code"
    
    # Add a special method to handle custom values
    @classmethod
    def _missing_(cls, value):
        """Handle missing enum values by creating them dynamically"""
        if isinstance(value, str):
            # Register the custom label
            cls.register_custom_labels([value])
            # Create a pseudo-enum member that behaves like a string but preserves the original value
            pseudo_member = str.__new__(cls, value)
            pseudo_member._name_ = value
            pseudo_member._value_ = value
            # Add it to the class so it can be accessed later
            setattr(cls, value, pseudo_member)
            logger.info(f"🔧 DYNAMIC LABEL: Created NodeLabel.{value} = '{value}'")
            return pseudo_member
        return None
    
    @classmethod
    def register_custom_labels(cls, custom_labels: List[str]):
        """Register custom node labels at runtime"""
        global _custom_labels_registry
        logger.info(f"🔧 REGISTRY DEBUG: cls={cls}, type(cls)={type(cls)}")
        logger.info(f"🔧 REGISTRY DEBUG: custom_labels={custom_labels}, type(custom_labels)={type(custom_labels)}")
        logger.info(f"🔧 REGISTRY DEBUG: _custom_labels_registry={_custom_labels_registry}, type={type(_custom_labels_registry)}")
        _custom_labels_registry.update(custom_labels)
        logger.info(f"🔧 REGISTRY: Registered custom node labels: {custom_labels}")
        logger.info(f"🔧 REGISTRY: Total custom labels: {list(_custom_labels_registry)}")
    
    @classmethod
    def clear_custom_labels(cls):
        """Clear all registered custom labels"""
        global _custom_labels_registry
        _custom_labels_registry.clear()
    
    @classmethod
    def get_system_labels(cls) -> List[str]:
        """Get list of system-defined node labels"""
        return [label.value for label in cls]
    
    @classmethod
    def get_custom_labels(cls) -> List[str]:
        """Get list of registered custom labels"""
        global _custom_labels_registry
        return list(_custom_labels_registry)
    
    @classmethod
    def is_system_label(cls, label: str) -> bool:
        """Check if label is a system-defined label"""
        return label in cls.get_system_labels()
    
    @classmethod
    def is_custom_label(cls, label: str) -> bool:
        """Check if label is a registered custom label"""
        global _custom_labels_registry
        return label in _custom_labels_registry
    
    @classmethod
    def get_all_valid_labels(cls, custom_labels: list = None) -> list:
        """Get all valid labels including system, registered custom, and provided custom labels"""
        all_labels = cls.get_system_labels()
        all_labels.extend(cls.get_custom_labels())  # Add registered custom labels
        if custom_labels:
            all_labels.extend(custom_labels)  # Add additional custom labels if provided
        return list(set(all_labels))  # Remove duplicates
    
    @classmethod
    def is_valid_label(cls, label: str, custom_labels: list = None) -> bool:
        """Check if label is valid (system, registered custom, or provided custom)"""
        return label in cls.get_all_valid_labels(custom_labels)
    

class RelationshipType(str, Enum):
    CREATED_BY = "CREATED_BY"
    WORKS_AT = "WORKS_AT"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    CONTAINS = "CONTAINS"
    ASSIGNED_TO = "ASSIGNED_TO"
    MANAGED_BY = "MANAGED_BY"
    RELATED_TO = "RELATED_TO"
    HAS = "HAS"
    IS_A = "IS_A"
    PARTICIPATED_IN = "PARTICIPATED_IN"
    BELONGS_TO = "BELONGS_TO"
    REPORTED_BY = "REPORTED_BY"
    REFERENCES = "REFERENCES"
    
    @classmethod
    def _missing_(cls, value):
        """Handle missing enum values by creating them dynamically"""
        if isinstance(value, str):
            # Register the custom relationship
            cls.register_custom_relationships([value])
            # Create a pseudo-enum member that behaves like a string but preserves the original value
            pseudo_member = str.__new__(cls, value)
            pseudo_member._name_ = value
            pseudo_member._value_ = value
            # Add it to the class so it can be accessed later
            setattr(cls, value, pseudo_member)
            logger.info(f"🔧 DYNAMIC RELATIONSHIP: Created RelationshipType.{value} = '{value}'")
            return pseudo_member
        return None
    
    @classmethod
    def register_custom_relationships(cls, custom_relationships: List[str]):
        """Register custom relationship types at runtime"""
        global _custom_relationships_registry
        _custom_relationships_registry.update(custom_relationships)
        logger.info(f"🔧 REGISTRY: Registered custom relationship types: {custom_relationships}")
        logger.info(f"🔧 REGISTRY: Total custom relationships: {list(_custom_relationships_registry)}")
    
    @classmethod
    def clear_custom_relationships(cls):
        """Clear all registered custom relationships"""
        global _custom_relationships_registry
        _custom_relationships_registry.clear()
    
    @classmethod
    def get_system_relationships(cls) -> List[str]:
        """Get list of system-defined relationship types"""
        return [rel.value for rel in cls]
    
    @classmethod
    def get_custom_relationships(cls) -> List[str]:
        """Get list of registered custom relationships"""
        global _custom_relationships_registry
        return list(_custom_relationships_registry)
    
    @classmethod
    def is_system_relationship(cls, relationship: str) -> bool:
        """Check if relationship is a system-defined relationship"""
        return relationship in cls.get_system_relationships()
    
    @classmethod
    def is_custom_relationship(cls, relationship: str) -> bool:
        """Check if relationship is a registered custom relationship"""
        global _custom_relationships_registry
        return relationship in _custom_relationships_registry
    
    @classmethod
    def get_all_valid_relationships(cls, custom_relationships: list = None) -> list:
        """Get all valid relationships including system, registered custom, and provided custom relationships"""
        all_relationships = cls.get_system_relationships()
        all_relationships.extend(cls.get_custom_relationships())  # Add registered custom relationships
        if custom_relationships:
            all_relationships.extend(custom_relationships)  # Add additional custom relationships if provided
        return list(set(all_relationships))  # Remove duplicates
    
    @classmethod
    def is_valid_relationship(cls, relationship: str, custom_relationships: list = None) -> bool:
        """Check if relationship is valid (system, registered custom, or provided custom)"""
        return relationship in cls.get_all_valid_relationships(custom_relationships)

class ContextItem(BaseModel):
    """Context item for memory request"""
    role: Literal["user", "assistant"]
    content: str

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

# --- MemoryMetadata ---
def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Recursively flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

AllowedCustomMetadataType = Union[str, int, float, bool, List[str]]

class PreferredProvider(str, Enum):
    """Preferred provider for document processing."""
    GEMINI = "gemini"
    TENSORLAKE = "tensorlake"
    REDUCTO = "reducto"
    AUTO = "auto"


class PropertyOverrideRule(BaseModel):
    """Property override rule with optional match conditions"""
    nodeLabel: str = Field(
        ...,
        min_length=1,
        description="Node type to apply overrides to (e.g., 'User', 'SecurityBehavior')"
    )
    match: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional conditions that must be met for override to apply. If not provided, applies to all nodes of this type"
    )
    set: Dict[str, Any] = Field(
        ...,
        description="Properties to set/override on matching nodes"
    )
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "examples": [
                {
                    "nodeLabel": "User",
                    "match": {"name": "Alice"},
                    "set": {"id": "user_alice_123", "role": "project_manager"}
                },
                {
                    "nodeLabel": "Note", 
                    "set": {"pageId": "pg_123", "archived": False}
                }
            ]
        }
    )

class MemoryMetadata(BaseModel):
    """Metadata for memory request"""
    hierarchical_structures: Optional[Union[str, List]] = Field(
        None, 
        description="Hierarchical structures to enable navigation from broad topics to specific ones"
    )
    createdAt: Optional[str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO datetime when the memory was created"
    )
    location: Optional[str] = None
    topics: Optional[List[str]] = None
    emoji_tags: Optional[List[str]] = Field(None, alias="emoji tags")
    emotion_tags: Optional[List[str]] = Field(None, alias="emotion tags")
    conversationId: Optional[str] = None
    sourceUrl: Optional[str] = None
    
    # Role and category as primary metadata fields (not custom metadata)
    role: Optional[MessageRole] = Field(
        None,
        description="Role that generated this memory (user or assistant)"
    )
    category: Optional[Union[UserMemoryCategory, AssistantMemoryCategory]] = Field(
        None,
        description="Memory category based on role. For users: preference, task, goal, fact, context. For assistants: skills, learning, task, goal, fact, context."
    )

    # DEPRECATED: Use request-level fields instead
    # These fields are kept for backwards compatibility but will be removed in v2
    user_id: Optional[str] = Field(
        default=None,
        deprecated=True,
        description="DEPRECATED: Use 'external_user_id' at request level instead. "
                   "This field will be removed in v2."
    )
    external_user_id: Optional[str] = Field(
        default=None,
        deprecated=True,
        description="DEPRECATED: Use 'external_user_id' at request level instead. "
                   "This field will be removed in v2."
    )

    # =========================================================================
    # INTERNAL: Granular ACL fields for vector store filtering
    # =========================================================================
    # These fields are auto-populated from memory_policy.acl and should NOT
    # be set directly by developers. They enable efficient filtering in Qdrant/Pinecone.
    # Developer-facing ACL should be set via memory_policy.acl at request level.
    external_user_read_access: Optional[List[str]] = Field(
        default_factory=list,
        description="INTERNAL: Auto-populated for vector store filtering. Use memory_policy.acl instead."
    )
    external_user_write_access: Optional[List[str]] = Field(
        default_factory=list,
        description="INTERNAL: Auto-populated for vector store filtering. Use memory_policy.acl instead."
    )
    user_read_access: Optional[List[str]] = Field(
        default_factory=list,
        description="INTERNAL: Auto-populated for vector store filtering. Use memory_policy.acl instead."
    )
    user_write_access: Optional[List[str]] = Field(
        default_factory=list,
        description="INTERNAL: Auto-populated for vector store filtering. Use memory_policy.acl instead."
    )
    workspace_read_access: Optional[List[str]] = Field(
        default_factory=list,
        description="INTERNAL: Auto-populated for vector store filtering. Use memory_policy.acl instead."
    )
    workspace_write_access: Optional[List[str]] = Field(
        default_factory=list,
        description="INTERNAL: Auto-populated for vector store filtering. Use memory_policy.acl instead."
    )
    role_read_access: Optional[List[str]] = Field(
        default_factory=list,
        description="INTERNAL: Auto-populated for vector store filtering. Use memory_policy.acl instead."
    )
    role_write_access: Optional[List[str]] = Field(
        default_factory=list,
        description="INTERNAL: Auto-populated for vector store filtering. Use memory_policy.acl instead."
    )
    namespace_read_access: Optional[List[str]] = Field(
        default_factory=list,
        description="INTERNAL: Auto-populated for vector store filtering. Use memory_policy.acl instead."
    )
    namespace_write_access: Optional[List[str]] = Field(
        default_factory=list,
        description="INTERNAL: Auto-populated for vector store filtering. Use memory_policy.acl instead."
    )
    organization_read_access: Optional[List[str]] = Field(
        default_factory=list,
        description="INTERNAL: Auto-populated for vector store filtering. Use memory_policy.acl instead."
    )
    organization_write_access: Optional[List[str]] = Field(
        default_factory=list,
        description="INTERNAL: Auto-populated for vector store filtering. Use memory_policy.acl instead."
    )

    pageId: Optional[str] = None
    sourceType: Optional[str] = None
    workspace_id: Optional[str] = None
    upload_id: Optional[str] = Field(None, description="Upload ID for document processing workflows")
    # Multi-tenant context (IDs only; pointers are set only in Parse payload)
    # DEPRECATED: Use request-level fields instead
    organization_id: Optional[str] = Field(
        default=None,
        deprecated=True,
        description="DEPRECATED: Use 'organization_id' at request level instead. "
                   "This field will be removed in v2."
    )
    namespace_id: Optional[str] = Field(
        default=None,
        deprecated=True,
        description="DEPRECATED: Use 'namespace_id' at request level instead. "
                   "This field will be removed in v2."
    )

    # =========================================================================
    # DEPRECATED: OMO fields - Use memory_policy at request level instead
    # =========================================================================
    # These fields are kept for backwards compatibility but will be removed in v2.
    # Set OMO safety standards via memory_policy.consent, memory_policy.risk,
    # and memory_policy.acl at request level.
    consent: Optional[str] = Field(
        default="implicit",
        deprecated=True,
        description="DEPRECATED: Use 'memory_policy.consent' at request level instead. "
                   "Values: 'explicit', 'implicit' (default), 'terms', 'none'."
    )
    risk: Optional[str] = Field(
        default="none",
        deprecated=True,
        description="DEPRECATED: Use 'memory_policy.risk' at request level instead. "
                   "Values: 'none' (default), 'sensitive', 'flagged'."
    )
    acl: Optional[Dict[str, List[str]]] = Field(
        default=None,
        deprecated=True,
        description="DEPRECATED: Use 'memory_policy.acl' at request level instead. "
                   "Format: {'read': [...], 'write': [...]}."
    )

    # QueryLog related fields
    sessionId: Optional[str] = None  # Session ID for tracking query context
    post: Optional[str] = None  # Post objectId pointer
    userMessage: Optional[str] = None  # PostMessage objectId pointer
    assistantMessage: Optional[str] = None  # PostMessage objectId pointer
    
    # Goal/UseCase/Step Classification (from client or generated)
    relatedGoals: Optional[List[str]] = Field(default_factory=list)  # Goal objectIds
    relatedUseCases: Optional[List[str]] = Field(default_factory=list)  # UseCase objectIds
    relatedSteps: Optional[List[str]] = Field(default_factory=list)  # Step objectIds
    goalClassificationScores: Optional[List[float]] = Field(default_factory=list)
    useCaseClassificationScores: Optional[List[float]] = Field(default_factory=list)
    stepClassificationScores: Optional[List[float]] = Field(default_factory=list)

    customMetadata: Optional[Dict[str, AllowedCustomMetadataType]] = Field(
        default=None,
        description="Optional object for arbitrary custom metadata fields. Only string, number, boolean, or list of strings allowed. Nested dicts are not allowed."
    )

    @field_validator('role', mode='before')
    @classmethod
    def validate_role(cls, v):
        """Convert string role values to MessageRole enum"""
        if v is None or isinstance(v, MessageRole):
            return v
        if isinstance(v, str):
            try:
                return MessageRole(v)
            except ValueError:
                raise ValueError(f"Invalid role '{v}'. Must be one of: {[r.value for r in MessageRole]}")
        return v

    @field_validator('category')
    @classmethod
    def validate_category_for_role(cls, v, info):
        """Validate that category matches the role"""
        if v is None:
            return v
            
        role = info.data.get('role')
        
        # Handle string values by converting to appropriate enum
        if isinstance(v, str):
            if role == MessageRole.USER:
                try:
                    return UserMemoryCategory(v)
                except ValueError:
                    raise ValueError(f"Invalid category '{v}' for user role. Must be one of: {[cat.value for cat in UserMemoryCategory]}")
            elif role == MessageRole.ASSISTANT:
                try:
                    return AssistantMemoryCategory(v)
                except ValueError:
                    raise ValueError(f"Invalid category '{v}' for assistant role. Must be one of: {[cat.value for cat in AssistantMemoryCategory]}")
            elif role is None:
                raise ValueError(f"Cannot validate category '{v}' without a role. Please provide 'role' field in metadata.")
        
        # Handle enum values - validate they match the role
        if isinstance(v, UserMemoryCategory) and role != MessageRole.USER:
            raise ValueError(f"UserMemoryCategory '{v.value}' cannot be used with role '{role}'")
        elif isinstance(v, AssistantMemoryCategory) and role != MessageRole.ASSISTANT:
            raise ValueError(f"AssistantMemoryCategory '{v.value}' cannot be used with role '{role}'")
        
        return v

    @field_validator("customMetadata")
    def validate_custom_metadata(cls, v):
        if v is None:
            return v
        for key, value in v.items():
            if isinstance(value, (str, int, float, bool)):
                continue
            elif isinstance(value, list) and all(isinstance(i, str) for i in value):
                continue
            else:
                raise ValueError(
                    f"customMetadata field '{key}' has invalid type {type(value).__name__}. "
                    "Only string, number, boolean, or list of strings are allowed. Nested dicts are not allowed."
                )
        return v

    @field_validator("topics", "emoji_tags", "emotion_tags", mode="before")
    @classmethod
    def ensure_list_of_strings(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            # Split on commas, strip whitespace, filter out empty
            return [item.strip() for item in v.split(",") if item.strip()]
        if isinstance(v, list):
            return v
        raise ValueError("Must be a list of strings or a comma-separated string")

    def flatten(self) -> Dict[str, Any]:
        """Return a flat dict of all metadata, including customMetadata. Handles legacy/str values for customMetadata and list fields."""
        base = self.model_dump(exclude_none=True)
        # Remove customMetadata from base to avoid double-inclusion
        custom = base.pop('customMetadata', {})
        # Robustly handle customMetadata legacy types
        if custom is None or custom == 'None':
            custom = {}
        elif isinstance(custom, str):
            try:
                custom = json.loads(custom)
                if not isinstance(custom, dict):
                    custom = {}
            except Exception:
                custom = {}
        # Flatten customMetadata (if present)
        flat_custom = flatten_dict(custom)
        # Ensure topics, emoji_tags, emotion_tags are always lists
        for field in ['topics', 'emoji_tags', 'emotion_tags']:
            val = base.get(field)
            if isinstance(val, str):
                base[field] = [item.strip() for item in val.split(',') if item.strip()]
            elif val is None:
                base[field] = []
        # Merge base and flattened custom
        return {**base, **flat_custom}

    def to_omo(
        self,
        memory_id: str,
        content: str,
        memory_type: str = "text",
        memory_policy: Optional[Any] = None
    ) -> "OpenMemoryObject":
        """
        Convert this MemoryMetadata to OMO (Open Memory Object) standard format.

        This enables memory portability across OMO-compliant platforms.
        Papr-specific fields are stored in ext.papr:* namespace.

        Args:
            memory_id: Unique memory identifier
            content: Memory content
            memory_type: Type (text, image, audio, video, file, code)
            memory_policy: Optional MemoryPolicy instance

        Returns:
            OpenMemoryObject in OMO v1 format

        Example:
            >>> metadata = MemoryMetadata(external_user_id="user_123", consent="explicit")
            >>> omo = metadata.to_omo("mem_abc", "Meeting notes...", "text")
            >>> print(omo.model_dump_json())
        """
        from models.omo import memory_to_omo
        return memory_to_omo(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            metadata=self,
            memory_policy=memory_policy
        )

    @classmethod
    def from_omo(cls, omo: "OpenMemoryObject") -> "MemoryMetadata":
        """
        Create MemoryMetadata from an OMO (Open Memory Object) standard format.

        This enables importing memories from other OMO-compliant platforms.

        Args:
            omo: OpenMemoryObject instance

        Returns:
            MemoryMetadata instance with fields populated from OMO

        Example:
            >>> omo = OpenMemoryObject(id="mem_123", content="...", consent="explicit", ...)
            >>> metadata = MemoryMetadata.from_omo(omo)
        """
        from models.omo import from_omo
        papr_data = from_omo(omo)
        metadata_dict = papr_data.get("metadata", {})
        return cls(**metadata_dict)

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        extra='allow'
    )

    def model_dump(self, *args, **kwargs):
        def convert_dt(obj):
            if isinstance(obj, dict):
                return {k: convert_dt(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dt(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_dt(v) for v in obj)
            elif isinstance(obj, set):
                return {convert_dt(v) for v in obj}
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj
        data = super().model_dump(*args, **kwargs)
        return convert_dt(data)


class UploadDocumentRequest(BaseModel):
    """Request model for uploading a document with metadata."""
    type: MemoryType = Field(
        default=MemoryType.DOCUMENT,
        description="Type of memory. Only 'document' is allowed."
    )
    metadata: Optional[MemoryMetadata] = Field(
        None,
        description="Metadata for the document upload, including user and ACL fields."
    )
    # Document processing specific fields
    preferred_provider: Optional[PreferredProvider] = Field(
        None,
        description="Preferred document processing provider (reducto, tensorlake, gemini, auto). If not provided, system will auto-select."
    )
    hierarchical_enabled: bool = Field(
        default=False,
        description="Enable hierarchical memory extraction for better organization of document content."
    )
    # Schema specification fields (from SchemaSpecificationMixin)
    schema_id: Optional[str] = Field(
        None,
        description="Schema ID to use for graph extraction. If not provided, system will auto-generate schema."
    )
    simple_schema_mode: bool = Field(
        default=False,
        description="Enable simple schema mode for faster processing with basic graph extraction."
    )
    graph_override: Optional[Dict[str, Any]] = Field(
        None,
        description="Override graph structure for custom schema enforcement. Not supported for documents (too complex for document processing)."
    )
    property_overrides: Optional[List[PropertyOverrideRule]] = Field(
        None,
        description="Property overrides for node customization with match conditions"
    )
    # Note: The file itself is handled by FastAPI's UploadFile, not Pydantic

    @field_validator("type")
    @classmethod
    def type_must_be_document(cls, v):
        if v != MemoryType.DOCUMENT:
            raise ValueError("type must be 'document'")
        return v

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        extra='forbid',
        json_schema_extra={
            "example": {
                "type": "document",
                "metadata": {
                    "user_id": "internal_user_123",
                    "external_user_id": "external_user_abc",
                    "external_user_read_access": ["external_user_abc", "external_user_xyz"],
                    "user_read_access": ["internal_user_123"],
                    "workspace_read_access": ["workspace_1"],
                    "role_read_access": ["admin"],
                    "customMetadata": {"source": "upload", "category": "report"}
                },
                "preferred_provider": "reducto",
                "hierarchical_enabled": True,
                "schema_id": "schema_123",
                "simple_schema_mode": False
            }
        }
    )

    

# OAuth2 Authentication Models
class LoginResponse(BaseModel):
    """Response model for OAuth2 login endpoint"""
    message: str = "Redirecting to Auth0 for authentication"
    redirect_url: Optional[str] = None

class TokenRequest(BaseModel):
    """Request model for OAuth2 token endpoint"""
    grant_type: str = Field(..., description="OAuth2 grant type (authorization_code, refresh_token)")
    code: Optional[str] = Field(None, description="Authorization code from OAuth2 callback")
    redirect_uri: Optional[str] = Field(None, description="Redirect URI used in authorization")
    client_type: Optional[str] = Field("papr_plugin", description="Client type (papr_plugin, browser_extension)")
    refresh_token: Optional[str] = Field(None, description="Refresh token for token refresh")

class TokenResponse(BaseModel):
    """Response model for OAuth2 token endpoint"""
    access_token: str = Field(..., description="OAuth2 access token")
    token_type: str = Field("Bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token for getting new access tokens")
    scope: str = Field(..., description="OAuth2 scopes granted")
    user_id: Optional[str] = Field(None, description="User ID from Auth0")
    message: Optional[str] = Field(None, description="Additional message or status")

class UserInfoResponse(BaseModel):
    """Response model for /me endpoint"""
    user_id: str = Field(..., description="Internal user ID")
    sessionToken: Optional[str] = Field(None, description="Session token for API access")
    imageUrl: Optional[str] = Field(None, description="User profile image URL")
    displayName: Optional[str] = Field(None, description="User display name")
    email: Optional[str] = Field(None, description="User email address")
    message: str = Field("You are authenticated!", description="Authentication status message")

class LogoutResponse(BaseModel):
    """Response model for logout endpoint"""
    message: str = Field("Redirecting to logout", description="Logout status message")
    logout_url: str = Field(..., description="URL to complete logout process")

class CallbackResponse(BaseModel):
    """Response model for OAuth2 callback endpoint"""
    message: str = Field("Authorization successful", description="Callback status message")
    code: Optional[str] = Field(None, description="Authorization code")
    state: Optional[str] = Field(None, description="State parameter for security")

class ErrorResponse(BaseModel):
    """Generic error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    code: int = Field(400, description="HTTP error code")



class FeedbackType(str, Enum):
    """Types of feedback that can be provided"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    CORRECTION = "correction"
    REPORT = "report"
    COPY_ACTION = "copy_action"
    SAVE_ACTION = "save_action"
    CREATE_DOCUMENT = "create_document"
    MEMORY_RELEVANCE = "memory_relevance"
    ANSWER_QUALITY = "answer_quality"

class FeedbackSource(str, Enum):
    """Where the feedback was provided from"""
    INLINE = "inline"
    POST_QUERY = "post_query"
    SESSION_END = "session_end"
    MEMORY_CITATION = "memory_citation"
    ANSWER_PANEL = "answer_panel"


# ============================================================================
# Memory Policy Models (Memory-Oriented Policies)
# ============================================================================
# These models provide a unified way to control how memories are processed,
# what graph nodes are created, and how they're constrained.

class PolicyMode(str, Enum):
    """
    Memory processing mode - describes WHO controls graph generation.

    - AUTO: LLM extracts entities freely (default)
    - MANUAL: Developer provides exact nodes (no LLM extraction)
    - HYBRID: LLM extracts with developer constraints

    Note: 'structured' is accepted as a deprecated alias for 'manual'.
    """
    AUTO = "auto"
    MANUAL = "manual"  # Renamed from STRUCTURED - developer provides exact nodes
    HYBRID = "hybrid"

    # DEPRECATED: Keep for backwards compatibility (maps to MANUAL)
    # Note: Can't have duplicate values in Enum, so we handle 'structured' via validation


class SearchMode(str, Enum):
    """Search mode for finding existing nodes."""
    SEMANTIC = "semantic"  # Vector similarity search
    EXACT = "exact"        # Exact property match
    FUZZY = "fuzzy"        # Partial/fuzzy match


class SearchConfig(BaseModel):
    """Configuration for finding existing nodes."""
    mode: SearchMode = Field(
        default=SearchMode.SEMANTIC,
        description="Search mode: 'semantic' (vector similarity), 'exact' (property match), 'fuzzy' (partial match)"
    )
    threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for semantic search (0.0-1.0)"
    )
    properties: Optional[List[str]] = Field(
        default=None,
        description="For exact/fuzzy mode: which properties to match on"
    )

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "examples": [
                {"mode": "semantic", "threshold": 0.85},
                {"mode": "exact", "properties": ["name", "email"]},
                {"mode": "fuzzy", "properties": ["name"], "threshold": 0.7}
            ]
        }
    )


class NodeConstraint(BaseModel):
    """
    Policy for how nodes of a specific type should be handled.

    Node constraints allow developers to control:
    - Which node types can be created vs. linked
    - How to find existing nodes for linking
    - What values to force or merge on nodes
    - When to apply the constraint (conditional)
    """
    # === WHAT ===
    node_type: str = Field(
        ...,
        min_length=1,
        description="Node type this constraint applies to (e.g., 'Task', 'Project', 'Person')"
    )

    # === WHEN (conditional application) ===
    when: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Condition for when this constraint applies. All conditions must match (AND logic). "
                   "Example: {'priority': 'high'} only applies to high-priority nodes.",
        alias="match"  # Backwards compatibility
    )

    # === CREATION POLICY ===
    create: Literal["auto", "never"] = Field(
        default="auto",
        description="'auto': Create if not found via search. 'never': Only link to existing nodes (controlled vocabulary)."
    )

    # === NODE SELECTION ===
    node_id: Optional[str] = Field(
        default=None,
        description="Direct: Skip search, use this exact node ID. Useful for linking to known nodes."
    )
    search: Optional[SearchConfig] = Field(
        default=None,
        description="How to find existing nodes for linking/updating."
    )

    # === VALUE POLICIES ===
    force: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Force these property values on ALL matching nodes (new or existing). "
                   "Always wins over AI extraction.",
        alias="set"  # Backwards compatibility
    )
    merge: Optional[List[str]] = Field(
        default=None,
        description="Update these properties from LLM extraction on EXISTING nodes only. "
                   "Has no effect on newly created nodes.",
        alias="update_on_match"  # Backwards compatibility (also accept 'update')
    )

    model_config = ConfigDict(
        extra='forbid',
        populate_by_name=True,  # Accept both field name and alias
        json_schema_extra={
            "examples": [
                {
                    "node_type": "Task",
                    "create": "never",
                    "search": {"mode": "semantic", "threshold": 0.85},
                    "merge": ["status", "assignee"]
                },
                {
                    "node_type": "Project",
                    "force": {"workspace_id": "ws_123", "team": "engineering"}
                },
                {
                    "node_type": "Person",
                    "create": "never",
                    "when": {"role": "team_member"}
                }
            ]
        }
    )


class NodeSpec(BaseModel):
    """
    Specification for a node in structured mode.

    Used when mode='structured' to define exact nodes to create.
    """
    id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for this node"
    )
    type: str = Field(
        ...,
        min_length=1,
        description="Node type/label (e.g., 'Transaction', 'Product', 'Person')"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Properties for this node"
    )

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "examples": [
                {
                    "id": "txn_12345",
                    "type": "Transaction",
                    "properties": {"amount": 5.50, "product": "Latte", "timestamp": "2026-01-21T10:30:00Z"}
                }
            ]
        }
    )


class RelationshipSpec(BaseModel):
    """
    Specification for a relationship in structured mode.

    Used when mode='structured' to define exact relationships between nodes.
    """
    source: str = Field(
        ...,
        min_length=1,
        description="ID of the source node"
    )
    target: str = Field(
        ...,
        min_length=1,
        description="ID of the target node"
    )
    type: str = Field(
        ...,
        min_length=1,
        description="Relationship type (e.g., 'PURCHASED', 'WORKS_AT', 'ASSIGNED_TO')"
    )
    properties: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional properties for this relationship"
    )

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "examples": [
                {
                    "source": "txn_12345",
                    "target": "product_latte",
                    "type": "PURCHASED"
                }
            ]
        }
    )


class MemoryPolicy(BaseModel):
    """
    Unified memory processing policy.

    This is the SINGLE source of truth for how a memory should be processed,
    combining graph generation control AND OMO (Open Memory Object) safety standards.

    **Graph Generation Modes:**
    - auto: LLM extracts entities freely (default)
    - manual: Developer provides exact nodes (no LLM extraction)
    - hybrid: LLM extracts with constraints

    Note: 'structured' is accepted as a deprecated alias for 'manual'.

    **OMO Safety Standards:**
    - consent: How data owner allowed storage (explicit, implicit, terms, none)
    - risk: Safety assessment (none, sensitive, flagged)
    - acl: Access control list for read/write permissions

    **Schema Integration:**
    - schema_id: Reference a schema that may have its own default memory_policy
    - Schema-level policies are merged with request-level (request takes precedence)
    """

    # =========================================================================
    # GRAPH GENERATION
    # =========================================================================

    mode: PolicyMode = Field(
        default=PolicyMode.AUTO,
        description="How to generate graph from this memory. "
                   "'auto': LLM extracts entities freely. "
                   "'manual': You provide exact nodes (no LLM). "
                   "'hybrid': LLM extracts with your constraints applied. "
                   "Note: 'structured' is accepted as deprecated alias for 'manual'."
    )

    # For MANUAL mode: Direct graph specification
    nodes: Optional[List[NodeSpec]] = Field(
        default=None,
        description="For manual mode: Exact nodes to create (no LLM extraction). "
                   "Required when mode='manual'. Each node needs id, type, and properties."
    )
    relationships: Optional[List[RelationshipSpec]] = Field(
        default=None,
        description="For manual mode: Exact relationships between nodes. "
                   "References node IDs defined in 'nodes' array."
    )

    # For AUTO/HYBRID mode: Node constraints (policies)
    node_constraints: Optional[List[NodeConstraint]] = Field(
        default=None,
        description="Rules for how LLM-extracted nodes should be created/updated. "
                   "Used in 'auto' and 'hybrid' modes. Controls creation policy, "
                   "property forcing, and merge behavior."
    )

    # Schema reference
    schema_id: Optional[str] = Field(
        default=None,
        description="Reference a UserGraphSchema by ID. The schema's memory_policy "
                   "(if defined) will be used as defaults, with this request's "
                   "settings taking precedence."
    )

    # Schema mode control (from deprecated GraphGeneration.auto.simple_schema_mode)
    simple_schema_mode: bool = Field(
        default=False,
        description="Limit AI extraction to system types + one user schema for consistency. "
                   "Use with schema_id to ensure predictable graph structure."
    )

    # =========================================================================
    # OMO SAFETY STANDARDS
    # =========================================================================

    consent: Optional[str] = Field(
        default="implicit",
        description="How the data owner allowed this memory to be stored/used. "
                   "'explicit': User explicitly agreed. "
                   "'implicit': Inferred from context (default). "
                   "'terms': Covered by Terms of Service. "
                   "'none': No consent - graph extraction will be SKIPPED."
    )

    risk: Optional[str] = Field(
        default="none",
        description="Safety assessment for this memory. "
                   "'none': Safe content (default). "
                   "'sensitive': Contains PII or sensitive info. "
                   "'flagged': Requires review - ACL will be restricted to owner only."
    )

    acl: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Access control list (ACL) for this memory and its graph nodes. "
                   "Conforms to Open Memory Object (OMO) standard. "
                   "Format: {'read': ['user_id_1', ...], 'write': ['user_id_1', ...]}. "
                   "If not provided, defaults based on external_user_id and developer. "
                   "See: https://github.com/anthropics/open-memory-object"
    )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    @field_validator('mode', mode='before')
    @classmethod
    def normalize_mode(cls, v):
        """
        Normalize mode value, accepting 'structured' as deprecated alias for 'manual'.
        """
        if v == 'structured':
            # Log deprecation warning (import logger at module level)
            import logging
            logging.getLogger(__name__).warning(
                "mode='structured' is deprecated, use mode='manual' instead. "
                "This alias will be removed in a future version."
            )
            return 'manual'
        return v

    @field_validator('consent')
    @classmethod
    def validate_consent(cls, v):
        """Validate consent level."""
        valid_levels = {"explicit", "implicit", "terms", "none"}
        if v and v not in valid_levels:
            raise ValueError(f"consent must be one of {valid_levels}, got '{v}'")
        return v

    @field_validator('risk')
    @classmethod
    def validate_risk(cls, v):
        """Validate risk level."""
        valid_levels = {"none", "sensitive", "flagged"}
        if v and v not in valid_levels:
            raise ValueError(f"risk must be one of {valid_levels}, got '{v}'")
        return v

    @model_validator(mode='after')
    def validate_mode_requirements(self):
        """Validate that mode-specific fields are properly set."""
        if self.mode == PolicyMode.MANUAL:
            if not self.nodes:
                raise ValueError(
                    "mode='manual' requires 'nodes' to be provided. "
                    "Use mode='auto' for LLM extraction or provide exact nodes."
                )
        return self

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "examples": [
                {
                    "name": "Auto Mode (Default)",
                    "summary": "LLM extracts entities freely",
                    "value": {
                        "mode": "auto"
                    }
                },
                {
                    "name": "Manual Mode with Exact Nodes",
                    "summary": "Developer provides exact graph structure",
                    "value": {
                        "mode": "manual",
                        "nodes": [
                            {"id": "txn_001", "type": "Transaction", "properties": {"amount": 5.50}}
                        ],
                        "relationships": [
                            {"source": "txn_001", "target": "prod_001", "type": "PURCHASED"}
                        ]
                    }
                },
                {
                    "name": "Hybrid Mode with Constraints",
                    "summary": "LLM extracts with your rules applied",
                    "value": {
                        "mode": "hybrid",
                        "node_constraints": [
                            {"node_type": "Task", "create": "never", "merge": ["status"]},
                            {"node_type": "Person", "create": "never"}
                        ]
                    }
                },
                {
                    "name": "With OMO Safety Settings",
                    "summary": "Explicit consent with restricted access",
                    "value": {
                        "mode": "auto",
                        "consent": "explicit",
                        "risk": "sensitive",
                        "acl": {"read": ["user_alice"], "write": ["user_alice"]}
                    }
                },
                {
                    "name": "Using Schema Defaults",
                    "summary": "Inherit policy from schema",
                    "value": {
                        "schema_id": "schema_project_mgmt_v1"
                    }
                }
            ]
        }
    )


# ============================================================================
# OMO (Open Memory Object) Safety Standards
# ============================================================================
# These models implement the Open Memory Object standard for consent,
# risk assessment, and access control.

class ConsentLevel(str, Enum):
    """
    How the data owner allowed this memory to be stored/used.

    Aligned with Open Memory Object (OMO) standard.
    """
    EXPLICIT = "explicit"   # User explicitly agreed to store
    IMPLICIT = "implicit"   # Inferred from usage context (default)
    TERMS = "terms"         # Covered by Terms of Service
    NONE = "none"          # No consent recorded


class RiskLevel(str, Enum):
    """
    Post-ingest safety assessment of memory content.

    Aligned with Open Memory Object (OMO) standard.
    """
    NONE = "none"           # Safe content (default)
    SENSITIVE = "sensitive" # Contains PII, financial, health info
    FLAGGED = "flagged"     # Requires human review before retrieval


class ACLConfig(BaseModel):
    """
    Simplified Access Control List configuration.

    Aligned with Open Memory Object (OMO) standard.
    See: https://github.com/anthropics/open-memory-object

    **Supported Entity Prefixes:**

    | Prefix | Description | Validation |
    |--------|-------------|------------|
    | `user:` | Internal Papr user ID | Validated against Parse users |
    | `external_user:` | Your app's user ID | Not validated (your responsibility) |
    | `organization:` | Organization ID | Validated against your organizations |
    | `namespace:` | Namespace ID | Validated against your namespaces |
    | `workspace:` | Workspace ID | Validated against your workspaces |
    | `role:` | Parse role ID | Validated against your roles |

    **Examples:**
    ```python
    acl = ACLConfig(
        read=["external_user:alice_123", "organization:org_acme"],
        write=["external_user:alice_123"]
    )
    ```

    **Validation Rules:**
    - Internal entities (user, organization, namespace, workspace, role) are validated
    - External entities (external_user) are NOT validated - your app is responsible
    - Invalid internal entities will return an error
    - Unprefixed values default to `external_user:` for backwards compatibility
    """

    # Supported entity prefixes for ACL
    INTERNAL_PREFIXES = {"user:", "organization:", "namespace:", "workspace:", "role:"}
    EXTERNAL_PREFIXES = {"external_user:"}
    ALL_PREFIXES = INTERNAL_PREFIXES | EXTERNAL_PREFIXES

    read: List[str] = Field(
        default_factory=list,
        description="Entity IDs that can read this memory. "
                   "Format: 'prefix:id' (e.g., 'external_user:alice', 'organization:org_123'). "
                   "Supported prefixes: user, external_user, organization, namespace, workspace, role. "
                   "Unprefixed values treated as external_user for backwards compatibility."
    )
    write: List[str] = Field(
        default_factory=list,
        description="Entity IDs that can write/modify this memory. "
                   "Format: 'prefix:id' (e.g., 'external_user:alice'). "
                   "Supported prefixes: user, external_user, organization, namespace, workspace, role."
    )

    @field_validator('read', 'write', mode='before')
    @classmethod
    def normalize_entity_ids(cls, v):
        """
        Normalize entity IDs to include prefix.
        Unprefixed values default to external_user: for backwards compatibility.
        """
        if not v:
            return v
        normalized = []
        for entity_id in v:
            if not isinstance(entity_id, str):
                continue
            # Check if already has a valid prefix
            has_prefix = any(entity_id.startswith(p) for p in cls.ALL_PREFIXES)
            if has_prefix:
                normalized.append(entity_id)
            else:
                # Default to external_user: for backwards compatibility
                normalized.append(f"external_user:{entity_id}")
        return normalized

    @model_validator(mode='after')
    def validate_entity_format(self):
        """
        Validate entity ID format.
        Internal entities will be validated at request time against the database.
        """
        all_entities = (self.read or []) + (self.write or [])
        for entity_id in all_entities:
            if ':' not in entity_id:
                raise ValueError(
                    f"Invalid ACL entity format: '{entity_id}'. "
                    f"Expected 'prefix:id' format. Supported prefixes: "
                    f"{', '.join(sorted(self.ALL_PREFIXES))}"
                )
            prefix = entity_id.split(':')[0] + ':'
            if prefix not in self.ALL_PREFIXES:
                raise ValueError(
                    f"Unknown ACL entity prefix: '{prefix}' in '{entity_id}'. "
                    f"Supported prefixes: {', '.join(sorted(self.ALL_PREFIXES))}"
                )
        return self

    def get_internal_entities(self) -> Dict[str, List[str]]:
        """
        Extract internal entities that need validation.
        Returns dict grouped by prefix: {'user': [...], 'organization': [...], ...}
        """
        result = {prefix.rstrip(':'): [] for prefix in self.INTERNAL_PREFIXES}
        for entity_id in (self.read or []) + (self.write or []):
            for prefix in self.INTERNAL_PREFIXES:
                if entity_id.startswith(prefix):
                    entity_type = prefix.rstrip(':')
                    entity_value = entity_id[len(prefix):]
                    result[entity_type].append(entity_value)
        return {k: v for k, v in result.items() if v}  # Remove empty

    def get_external_entities(self) -> List[str]:
        """
        Extract external entities (not validated by Papr).
        """
        result = []
        for entity_id in (self.read or []) + (self.write or []):
            if entity_id.startswith("external_user:"):
                result.append(entity_id[len("external_user:"):])
        return result

    def to_granular_acl(self) -> Dict[str, List[str]]:
        """
        Convert simplified ACL to granular ACL fields for vector store filtering.
        Returns dict with keys like 'user_read_access', 'organization_read_access', etc.
        """
        granular = {
            'user_read_access': [],
            'user_write_access': [],
            'external_user_read_access': [],
            'external_user_write_access': [],
            'organization_read_access': [],
            'organization_write_access': [],
            'namespace_read_access': [],
            'namespace_write_access': [],
            'workspace_read_access': [],
            'workspace_write_access': [],
            'role_read_access': [],
            'role_write_access': [],
        }

        prefix_map = {
            'user:': 'user',
            'external_user:': 'external_user',
            'organization:': 'organization',
            'namespace:': 'namespace',
            'workspace:': 'workspace',
            'role:': 'role',
        }

        for entity_id in (self.read or []):
            for prefix, key in prefix_map.items():
                if entity_id.startswith(prefix):
                    granular[f'{key}_read_access'].append(entity_id[len(prefix):])
                    break

        for entity_id in (self.write or []):
            for prefix, key in prefix_map.items():
                if entity_id.startswith(prefix):
                    granular[f'{key}_write_access'].append(entity_id[len(prefix):])
                    break

        return granular

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "examples": [
                {
                    "description": "Share with specific user and entire organization",
                    "read": ["external_user:alice_123", "organization:org_acme"],
                    "write": ["external_user:alice_123"]
                },
                {
                    "description": "Namespace-scoped access",
                    "read": ["namespace:ns_production", "external_user:admin_bob"],
                    "write": ["external_user:admin_bob"]
                }
            ]
        }
    )


class OMOFilter(BaseModel):
    """
    Filter for Open Memory Object (OMO) safety standards in search/retrieval.

    Use this to filter search results by consent level and/or risk level.
    """
    min_consent: Optional[ConsentLevel] = Field(
        default=None,
        description="Minimum consent level required. Excludes memories with lower consent levels. "
                   "Order: explicit > implicit > terms > none. "
                   "Example: min_consent='implicit' excludes 'none' consent memories."
    )
    exclude_consent: Optional[List[ConsentLevel]] = Field(
        default=None,
        description="Explicitly exclude memories with these consent levels. "
                   "Example: exclude_consent=['none'] filters out all memories without consent."
    )
    max_risk: Optional[RiskLevel] = Field(
        default=None,
        description="Maximum risk level allowed. Excludes memories with higher risk. "
                   "Order: none < sensitive < flagged. "
                   "Example: max_risk='none' excludes 'sensitive' and 'flagged' memories."
    )
    exclude_risk: Optional[List[RiskLevel]] = Field(
        default=None,
        description="Explicitly exclude memories with these risk levels. "
                   "Example: exclude_risk=['flagged'] filters out all flagged content."
    )
    require_consent: bool = Field(
        default=False,
        description="If true, only return memories with explicit consent (consent != 'none'). "
                   "Shorthand for exclude_consent=['none']."
    )
    exclude_flagged: bool = Field(
        default=False,
        description="If true, exclude all flagged content (risk == 'flagged'). "
                   "Shorthand for exclude_risk=['flagged']."
    )

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "examples": [
                {"require_consent": True, "exclude_flagged": True},
                {"min_consent": "implicit", "max_risk": "sensitive"},
                {"exclude_consent": ["none"], "exclude_risk": ["flagged"]}
            ]
        }
    )