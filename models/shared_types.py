# Shared types for memory, parse_server, and structured_outputs
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum
import logging
from datetime import datetime, timezone, UTC
logger = logging.getLogger(__name__)
import json
 

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
    hierarchical_structures: Optional[str] = Field(
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

    user_id: Optional[str] = None
    external_user_id: Optional[str] = None

    # ACL fields for fine-grained access control
    external_user_read_access: Optional[List[str]] = Field(default_factory=list)
    external_user_write_access: Optional[List[str]] = Field(default_factory=list)
    user_read_access: Optional[List[str]] = Field(default_factory=list)
    user_write_access: Optional[List[str]] = Field(default_factory=list)
    workspace_read_access: Optional[List[str]] = Field(default_factory=list)
    workspace_write_access: Optional[List[str]] = Field(default_factory=list)
    role_read_access: Optional[List[str]] = Field(default_factory=list)
    role_write_access: Optional[List[str]] = Field(default_factory=list)
    namespace_read_access: Optional[List[str]] = Field(default_factory=list)
    namespace_write_access: Optional[List[str]] = Field(default_factory=list)
    organization_read_access: Optional[List[str]] = Field(default_factory=list)
    organization_write_access: Optional[List[str]] = Field(default_factory=list)

    pageId: Optional[str] = None
    sourceType: Optional[str] = None
    workspace_id: Optional[str] = None
    upload_id: Optional[str] = Field(None, description="Upload ID for document processing workflows")
    # Multi-tenant context (IDs only; pointers are set only in Parse payload)
    organization_id: Optional[str] = None
    namespace_id: Optional[str] = None

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