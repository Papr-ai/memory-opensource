from pydantic import BaseModel, Field, ConfigDict, field_validator, field_serializer, model_validator
from typing import Optional, List, Dict, Any, Literal, TypedDict, Union
from datetime import datetime, timezone, UTC
from enum import Enum
import json
from services.logging_config import get_logger
from models.shared_types import MemoryMetadata, ContextItem
from services.context_utils import parse_context
from models.shared_types import FeedbackType, FeedbackSource


logger = get_logger(__name__)

# ============================================================================
# MULTI-TENANT ENUMS
# ============================================================================

class UserType(str, Enum):
    """User type classification for multi-tenant architecture"""
    DEVELOPER = "DEVELOPER"
    END_USER = "END_USER"
    TEAM_MEMBER = "TEAM_MEMBER"

class EnvironmentType(str, Enum):
    """Environment types for namespaces"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

# ============================================================================
# PARSE BASE CLASSES
# ============================================================================

class ParsePointer(BaseModel):
    """A pointer to a Parse object"""
    objectId: str
    type: str = Field(default="Pointer", alias="__type")  # Using Field with alias
    className: str

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='forbid',
        alias_generator=None
    )

    def model_dump(self, *args, **kwargs):
        """Override model_dump to ensure __type is included"""
        data = super().model_dump(*args, **kwargs)
        # Ensure the type field is output as __type
        if 'type' in data:
            data['__type'] = data.pop('type')
        return data

# ============================================================================
# MULTI-TENANT POINTER CLASSES
# ============================================================================

class OrganizationPointer(BaseModel):
    """Pointer to Organization object"""
    objectId: str
    type: str = Field(default="Pointer", alias="__type")
    className: Literal["Organization"] = "Organization"

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='forbid',
        alias_generator=None
    )

    def model_dump(self, *args, **kwargs):
        """Override model_dump to ensure __type is included"""
        data = super().model_dump(*args, **kwargs)
        if 'type' in data:
            data['__type'] = data.pop('type')
        return data

class NamespacePointer(BaseModel):
    """Pointer to Namespace object"""
    objectId: str
    type: str = Field(default="Pointer", alias="__type")
    className: Literal["Namespace"] = "Namespace"

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='forbid',
        alias_generator=None
    )

    def model_dump(self, *args, **kwargs):
        """Override model_dump to ensure __type is included"""
        data = super().model_dump(*args, **kwargs)
        if 'type' in data:
            data['__type'] = data.pop('type')
        return data

# ============================================================================
# MULTI-TENANT MODELS
# ============================================================================

class WorkspacePointer(BaseModel):
    """Pointer to WorkSpace object"""
    objectId: str
    type: str = Field(default="Pointer", alias="__type")
    className: Literal["WorkSpace"] = "WorkSpace"

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='forbid',
        alias_generator=None
    )

    def model_dump(self, *args, **kwargs):
        """Override model_dump to ensure __type is included"""
        data = super().model_dump(*args, **kwargs)
        if 'type' in data:
            data['__type'] = data.pop('type')
        return data

class SubscriptionPointer(BaseModel):
    """Pointer to Subscription object (Stripe)"""
    objectId: str
    type: str = Field(default="Pointer", alias="__type")
    className: Literal["Subscription"] = "Subscription"

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='forbid',
        alias_generator=None
    )

    def model_dump(self, *args, **kwargs):
        """Override model_dump to ensure __type is included"""
        data = super().model_dump(*args, **kwargs)
        if 'type' in data:
            data['__type'] = data.pop('type')
        return data

class ParseUserPointer(BaseModel):
    """A pointer to a Parse User object that can also handle expanded user objects"""
    objectId: str
    type: str = Field(default="Pointer", alias="__type")
    className: Literal["_User"] = "_User"

    # Add optional fields that might come from Parse
    displayName: Optional[str] = None
    fullname: Optional[str] = None
    profileimage: Optional[str] = None
    title: Optional[str] = None
    isOnline: Optional[bool] = None
    companies: Optional[Dict[str, Any]] = None
    
    # Multi-tenant fields (NEW) with Parse Pointers
    user_type: Optional[UserType] = None
    organization: Optional[OrganizationPointer] = Field(default=None, description="Organization (Pointer) if DEVELOPER or TEAM_MEMBER")
    developer_organization: Optional[OrganizationPointer] = Field(default=None, description="Organization (Pointer) if END_USER")
    external_id: Optional[str] = Field(default=None, description="Developer's ID for this end user")
    
    # Deprecated (kept for backward compatibility)
    isDeveloper: Optional[bool] = None
    organization_id: Optional[str] = Field(default=None, description="DEPRECATED: Use organization pointer instead")
    developer_organization_id: Optional[str] = Field(default=None, description="DEPRECATED: Use developer_organization pointer instead")

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='allow',  # Changed from 'forbid' to 'allow' for flexibility
        alias_generator=None
    )

    @field_validator('profileimage')
    @classmethod
    def validate_profileimage(cls, v):
        if v is None:
            return None
        # If it's already a string (URL), return it
        if isinstance(v, str):
            return v
        # If it's a Parse File object, return the URL
        if isinstance(v, dict):
            return v.get('url', None)
        return None
        
    def model_dump(self, *args, **kwargs):
        """Override model_dump to ensure __type is included"""
        data = super().model_dump(*args, **kwargs)
        # Ensure the type field is output as __type
        if 'type' in data:
            data['__type'] = data.pop('type')
        return data

# ============================================================================
# MULTI-TENANT MODELS
# ============================================================================

class Organization(BaseModel):
    """Organization model - represents a tenant/company using Papr"""
    objectId: Optional[str] = Field(default=None, description="Parse objectId, auto-generated if not provided")
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    
    # Core fields with Parse Pointers
    name: str = Field(..., description="Organization name")
    owner: ParseUserPointer = Field(..., description="Owner of the organization (Pointer to _User)")
    workspace: Optional[WorkspacePointer] = Field(default=None, description="Link to WorkSpace for backward compatibility")
    
    # Subscription & Billing (Parse Pointer)
    subscription: Optional[SubscriptionPointer] = Field(default=None, description="Stripe subscription (Pointer)")
    plan_tier: str = Field(default="developer", description="Subscription tier: developer (free), starter, growth, enterprise")

    # team members pointers
    team_members: Optional[List[ParseUserPointer]] = Field(default=None, description="Team members (Pointers to _User)")
    
    # Default rate limits at organization level (Developer tier defaults)
    rate_limits: Dict[str, Optional[int]] = Field(
        default_factory=lambda: {
            "max_memory_operations_per_month": 1000,  # 1K memory operations (Developer tier)
            "max_storage_gb": 1,  # 1GB storage
            "max_active_memories": 2500,  # 2,500 active memories
            "rate_limit_per_minute": 10
        },
        description="Organization-level rate limits (based on plan_tier)"
    )
    
    # Settings with Parse Pointers
    default_namespace: Optional[NamespacePointer] = Field(default=None, description="Default namespace (Pointer)")
    
    # ACL
    ACL: Optional[Dict[str, Dict[str, bool]]] = Field(default_factory=dict)
    
    # Parse Relations (many-to-many)
    # Note: Relations are handled differently - they're not included in the model dump
    # team_members: Relation<_User>  # Handled via Parse REST API
    # allowed_namespaces: Relation<Namespace>  # Handled via Parse REST API

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='allow'
    )

    def model_dump(self, *args, **kwargs):
        """Override model_dump to handle datetime serialization"""
        def convert_dt(obj):
            if isinstance(obj, dict):
                return {k: convert_dt(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dt(v) for v in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        data = super().model_dump(*args, **kwargs)
        return convert_dt(data)

class Namespace(BaseModel):
    """Namespace model - isolated environment within an organization"""
    objectId: Optional[str] = Field(default=None, description="Parse objectId, auto-generated if not provided")
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    
    # Core fields with Parse Pointers
    name: str = Field(..., description="Namespace name (e.g., 'acme-production')")
    organization: OrganizationPointer = Field(..., description="Organization this namespace belongs to (Pointer)")
    environment_type: EnvironmentType = Field(
        default=EnvironmentType.PRODUCTION,
        description="Environment type: development, staging, production"
    )
    
    # Status
    is_active: bool = Field(default=True, description="Whether this namespace is active")
    
    # Rate limits (can override organization defaults)
    rate_limits: Dict[str, Optional[int]] = Field(
        default_factory=lambda: {
            "max_memory_operations_per_month": None,  # None = inherit from org
            "max_storage_gb": None,  # None = inherit from org
            "max_active_memories": None,  # None = inherit from org
            "rate_limit_per_minute": None  # None = inherit from org
        },
        description="Rate limits for this namespace (None = inherit from organization)"
    )
    
    # ACL
    ACL: Optional[Dict[str, Dict[str, bool]]] = Field(default_factory=dict)

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='allow'
    )

    def model_dump(self, *args, **kwargs):
        """Override model_dump to handle datetime serialization"""
        def convert_dt(obj):
            if isinstance(obj, dict):
                return {k: convert_dt(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dt(v) for v in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        data = super().model_dump(*args, **kwargs)
        return convert_dt(data)

class APIKey(BaseModel):
    """API Key model - for authenticating to a specific namespace"""
    objectId: Optional[str] = Field(default=None, description="Parse objectId, auto-generated if not provided")
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    
    # Core fields
    key: str = Field(..., description="The actual API key")
    name: str = Field(..., description="Human-readable name for this key (e.g., 'Production Key')")
    
    # Tenant hierarchy with Parse Pointers
    namespace: NamespacePointer = Field(..., description="Namespace this key belongs to (Pointer)")
    organization: OrganizationPointer = Field(..., description="Organization (Pointer, for quick lookup)")
    
    # Metadata
    environment: str = Field(default="production", description="Environment: production, development, staging")
    permissions: List[str] = Field(
        default_factory=lambda: ["read", "write", "delete"],
        description="Permissions granted to this key"
    )
    
    # Status
    is_active: bool = Field(default=True, description="Whether this key is active")
    last_used_at: Optional[datetime] = Field(default=None, description="Last time this key was used")
    
    # ACL
    ACL: Optional[Dict[str, Dict[str, bool]]] = Field(default_factory=dict)

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='allow'
    )

    def model_dump(self, *args, **kwargs):
        """Override model_dump to handle datetime serialization"""
        def convert_dt(obj):
            if isinstance(obj, dict):
                return {k: convert_dt(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dt(v) for v in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        data = super().model_dump(*args, **kwargs)
        return convert_dt(data)

# ============================================================================
# MEMORY API RESPONSE MODELS
# ============================================================================

class AddMemoryItem(BaseModel):
    """Response model for a single memory item in add_memory response"""
    memoryId: str
    createdAt: datetime
    objectId: str
    memoryChunkIds: List[str] = Field(default_factory=list)
    
    @field_validator('memoryChunkIds', mode='before')
    @classmethod
    def validate_memory_chunk_ids(cls, v) -> List[str]:
        logger.info(f"Validating memoryChunkIds input: {v} of type {type(v)}")
        if v is None:
            logger.warning("memoryChunkIds is None, returning empty list")
            return []
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    logger.info(f"Parsed memoryChunkIds from JSON string: {parsed}")
                    return [str(x) for x in parsed if x]
            except json.JSONDecodeError:
                # If it's a comma-separated string
                if ',' in v:
                    chunks = [x.strip() for x in v.split(',')]
                    logger.info(f"Split memoryChunkIds from comma-separated string: {chunks}")
                    return [x for x in chunks if x]
                # If it's a single value
                if v.strip():
                    logger.info(f"Single memoryChunkId from string: {[v.strip()]}")
                    return [v.strip()]
        if isinstance(v, list):
            logger.info(f"memoryChunkIds is already a list: {v}")
            return [str(x) for x in v if x]
        logger.warning(f"Unexpected memoryChunkIds type: {type(v)}, returning empty list")
        return []

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='forbid'
    )

    @field_serializer('createdAt', when_used='json')
    def serialize_created_at(self, v):
        return v.isoformat() if v else None

class AddMemoryResponse(BaseModel):
    """Unified response model for add_memory API endpoint (success or error)."""
    code: int = Field(default=200, description="HTTP status code")
    status: str = Field(default="success", description="'success' or 'error'")
    data: Optional[List[AddMemoryItem]] = Field(default=None, description="List of memory items if successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    details: Optional[Any] = Field(default=None, description="Additional error details or context")

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

    def model_dump(self, *args, **kwargs):
        """Override model_dump to ensure nested datetime serialization"""
        d = super().model_dump(*args, **kwargs)
        if d.get('data'):
            for item in d['data']:
                if isinstance(item.get('createdAt'), datetime):
                    item['createdAt'] = item['createdAt'].isoformat()
        return d

    @classmethod
    def success(cls, data: List[AddMemoryItem], code: int = 200):
        return cls(code=code, status="success", data=data, error=None, details=None)

    @classmethod
    def failure(cls, error: str, code: int = 400, details: Any = None):
        return cls(code=code, status="error", data=None, error=error, details=details)


class AddMemoryOMOResponse(BaseModel):
    """
    OMO (Open Memory Object) format response for add_memory API.

    Used when ?format=omo is specified. Returns the memory in portable OMO format
    as defined by https://github.com/papr-ai/open-memory-object
    """
    code: int = Field(default=200, description="HTTP status code")
    status: str = Field(default="success", description="'success' or 'error'")
    omo: Optional[Any] = Field(default=None, description="OpenMemoryObject in OMO v1 format")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='allow'
    )

    @classmethod
    def success(cls, omo_object: Any, code: int = 200):
        """Create success response with OMO object."""
        return cls(code=code, status="success", omo=omo_object, error=None)

    @classmethod
    def failure(cls, error: str, code: int = 400):
        """Create failure response."""
        return cls(code=code, status="error", omo=None, error=error)


class ErrorDetail(BaseModel):
    code: int
    detail: str

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

class MemoryErrorResponse(BaseModel):
    status_code: int
    success: bool = False
    error: str
    data: Optional[Any] = None

class BatchMemoryError(BaseModel):
    index: int
    error: str
    code: Optional[int] = None
    status: Optional[str] = None
    details: Optional[Any] = None

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

class BatchMemoryResponse(BaseModel):
    code: int = Field(default=200, description="HTTP status code for the batch operation")
    status: str = Field(default="success", description="'success', 'partial', or 'error'")
    message: Optional[str] = Field(default=None, description="Human-readable status message")
    error: Optional[str] = Field(default=None, description="Batch-level error message, if any")
    details: Optional[Any] = Field(default=None, description="Additional error details or context")
    successful: List[AddMemoryResponse] = Field(default_factory=list, description="List of successful add responses")
    errors: List[BatchMemoryError] = Field(default_factory=list, description="List of errors for failed items")
    total_processed: int = 0
    total_successful: int = 0
    total_failed: int = 0
    total_content_size: int = 0
    total_storage_size: int = 0

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

    def model_dump(self, *args, **kwargs):
        """Override model_dump to ensure nested datetime serialization"""
        d = super().model_dump(*args, **kwargs)
        if d.get('successful'):
            for response in d['successful']:
                if response.get('data'):
                    for item in response['data']:
                        if isinstance(item.get('createdAt'), datetime):
                            item['createdAt'] = item['createdAt'].isoformat()
        return d

    @classmethod
    def success(cls, successful, **kwargs):
        # Only set these if not already in kwargs
        if 'total_processed' not in kwargs:
            kwargs['total_processed'] = len(successful)
        if 'total_successful' not in kwargs:
            kwargs['total_successful'] = len(successful)
        if 'total_failed' not in kwargs:
            kwargs['total_failed'] = 0
        return cls(
            code=200,
            status="success",
            successful=successful,
            errors=[],
            **kwargs
        )

    @classmethod
    def partial(cls, successful, errors, **kwargs):
        if 'total_processed' not in kwargs:
            kwargs['total_processed'] = len(successful) + len(errors)
        if 'total_successful' not in kwargs:
            kwargs['total_successful'] = len(successful)
        if 'total_failed' not in kwargs:
            kwargs['total_failed'] = len(errors)
        return cls(
            code=207,
            status="partial",
            successful=successful,
            errors=errors,
            **kwargs
        )

    @classmethod
    def failure(cls, errors, code=400, error=None, details=None, **kwargs):
        return cls(
            code=code,
            status="error",
            successful=[],
            errors=errors,
            error=error,
            details=details,
            total_processed=len(errors),
            total_successful=0,
            total_failed=len(errors),
            **kwargs
        )

class DeveloperUserPointer(BaseModel):
    objectId: str
    type: str = Field(default="Pointer", alias="__type")
    className: Literal["DeveloperUser"] = "DeveloperUser"
    external_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    email: Optional[str] = None
    
    # Multi-tenant fields (NEW) with Parse Pointers
    organization: Optional[OrganizationPointer] = Field(default=None, description="Organization (Pointer) this end user belongs to")
    namespace: Optional[NamespacePointer] = Field(default=None, description="Namespace (Pointer) this end user belongs to")
    
    # Deprecated (kept for backward compatibility)
    organization_id: Optional[str] = Field(default=None, description="DEPRECATED: Use organization pointer instead")
    namespace_id: Optional[str] = Field(default=None, description="DEPRECATED: Use namespace pointer instead")

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='allow',
        alias_generator=None
    )

    @classmethod
    def model_validate(cls, data):
        # If metadata contains email, surface it at the top level
        if isinstance(data, dict):
            meta = data.get("metadata")
            if meta and isinstance(meta, dict) and "email" in meta:
                data["email"] = meta["email"]
        return super().model_validate(data)

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
    
class ParseStoredMemory(BaseModel):
    objectId: str
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    ACL: Dict[str, Dict[str, bool]]
    content: str
    metadata: Optional[Union[str, Dict[str, Any]]] = None
    sourceType: str = "papr"
    context: Optional[List[ContextItem]] = Field(default_factory=list)
    title: Optional[str] = None
    location: Optional[str] = None
    emojiTags: List[str] = Field(default_factory=list)
    hierarchicalStructures: str = ""
    type: str
    sourceUrl: str = ""
    conversationId: str = ""
    # Role and category as primary fields in Parse Server
    role: Optional[str] = Field(default=None, description="Role that generated this memory (user or assistant)")
    category: Optional[str] = Field(default=None, description="Memory category based on role")
    memoryId: str
    topics: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    current_step: Optional[str] = None
    memoryChunkIds: List[str] = Field(default_factory=list)
    user: Optional[ParseUserPointer] = None  # Made optional for backward compatibility with legacy data
    developerUser: Optional[DeveloperUserPointer] = None
    workspace: Optional[ParsePointer] = None
    post: Optional[ParsePointer] = None
    postMessage: Optional[ParsePointer] = None
    matchingChunkIds: Optional[List[str]] = None
    
    # Multi-tenant fields (NEW) with Parse Pointers
    organization: Optional[OrganizationPointer] = Field(default=None, description="Organization (Pointer) that owns this memory")
    namespace: Optional[NamespacePointer] = Field(default=None, description="Namespace (Pointer) this memory belongs to")
    
    # Deprecated (kept for backward compatibility)
    organization_id: Optional[str] = Field(default=None, description="DEPRECATED: Use organization pointer instead")
    namespace_id: Optional[str] = Field(default=None, description="DEPRECATED: Use namespace pointer instead")

    # Document specific fields
    page_number: Optional[int] = None
    total_pages: Optional[int] = None
    upload_id: Optional[str] = None
    extract_mode: Optional[str] = None
    file_url: Optional[str] = None
    filename: Optional[str] = None
    page: Optional[str] = None

    customMetadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional object for arbitrary custom metadata fields. Use this for new custom fields. Enables direct filtering in Parse."
    )

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

    # =============================================================================
    # RELEVANCE SCORES - Research-backed scoring system
    # =============================================================================
    # References:
    #   - BM25/IR: log1p() normalization for frequency (prevents popularity bias)
    #   - RRF (Cormack et al. 2009): score = Σ 1/(k + rank_i) for rank fusion
    #   - Time decay: exp(-λ * age) with configurable half-life
    #   - Multi-signal fusion outperforms single signals (LTR research)
    # =============================================================================

    # COMPUTED AT SEARCH TIME - Base signals:
    similarity_score: Optional[float] = Field(
        default=None,
        description="Cosine similarity from vector search (0-1). Measures semantic relevance to query."
    )
    popularity_score: Optional[float] = Field(
        default=None,
        description="Popularity signal (0-1): 0.5*cacheConfidenceWeighted30d + 0.5*citationConfidenceWeighted30d. Uses stored EMA fields."
    )
    recency_score: Optional[float] = Field(
        default=None,
        description="Recency signal (0-1): exp(-0.05 * days_since_last_access). Half-life ~14 days."
    )

    # ONLY when rank_results=True - Reranker signals:
    reranker_score: Optional[float] = Field(
        default=None,
        description="Reranker relevance (0-1). From cross-encoder (Cohere/Qwen3/BGE) or LLM (GPT-5-nano)."
    )
    reranker_confidence: Optional[float] = Field(
        default=None,
        description="Reranker confidence (0-1). Meaningful for LLM reranking; equals reranker_score for cross-encoders."
    )
    reranker_type: Optional[str] = Field(
        default=None,
        description="Reranker type: 'cross_encoder' (Cohere/Qwen3/BGE) or 'llm' (GPT-5-nano/GPT-4o-mini)."
    )

    # FINAL COMBINED SCORE:
    relevance_score: Optional[float] = Field(
        default=None,
        description="Final relevance (0-1). rank_results=False: 0.6*sim + 0.25*pop + 0.15*recency. rank_results=True: RRF-based fusion."
    )

    model_config = ConfigDict(
        from_attributes=True,  # Allows conversion from ORM objects
        validate_assignment=True,  # Validate during assignment
        populate_by_name=True,  # Allow population by field name as well as alias
        str_strip_whitespace=True,  # Strip whitespace from strings
        extra='allow'  # Allow extra attributes
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

    def without_metadata(self) -> 'ParseStoredMemory':
        """Create a copy of the memory item without metadata field."""
        data = self.model_dump(exclude={'metadata'})

        # Preserve org/namespace IDs if they only exist in metadata.
        metadata = self.metadata
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        if isinstance(metadata, dict):
            if not data.get('organization_id'):
                data['organization_id'] = metadata.get('organization_id')
            if not data.get('namespace_id'):
                data['namespace_id'] = metadata.get('namespace_id')

        # Preserve org/namespace IDs if only pointer fields are present.
        def _normalize_pointer_id(value):
            if value is None:
                return None
            if isinstance(value, str):
                return value or None
            if isinstance(value, dict):
                if not value:
                    return None
                return value.get('objectId') or value.get('id')
            return getattr(value, 'objectId', None) or getattr(value, 'id', None)

        if not data.get('organization_id'):
            data['organization_id'] = _normalize_pointer_id(getattr(self, 'organization', None))
        if not data.get('namespace_id'):
            data['namespace_id'] = _normalize_pointer_id(getattr(self, 'namespace', None))

        return ParseStoredMemory.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict) -> 'ParseStoredMemory':
        logger.info(f"ParseStoredMemory.from_dict - Input data: {data}")
        
        # If metadata is a string, try to parse it
        if isinstance(data.get('metadata'), str):
            try:
                metadata = json.loads(data['metadata'])
                logger.info(f"ParseStoredMemory.from_dict - Parsed metadata: {metadata}")
                # Check if memoryChunkIds exists in metadata
                if 'memoryChunkIds' in metadata:
                    logger.info(f"ParseStoredMemory.from_dict - Found memoryChunkIds in metadata: {metadata['memoryChunkIds']}")
                    # Make sure to use these IDs
                    data['memoryChunkIds'] = metadata['memoryChunkIds']
            except json.JSONDecodeError:
                logger.error(f"ParseStoredMemory.from_dict - Failed to parse metadata string: {data['metadata']}")
        
        logger.info(f"ParseStoredMemory.from_dict - Final data before conversion: {data}")
        # Ensure user is a ParseUserPointer if possible
        user = data.get('user')
        if isinstance(user, dict) and 'objectId' in user:
            data['user'] = ParseUserPointer(**user)
        # Ensure lists and dicts are not None
        for field in [
            'user_read_access', 'user_write_access', 'external_user_read_access', 'external_user_write_access',
            'workspace_read_access', 'workspace_write_access', 'role_read_access', 'role_write_access'
        ]:
            if data.get(field) is None:
                data[field] = []
        if data.get('ACL') is None:
            data['ACL'] = {}
        # Provide default type if missing (for backward compatibility with old data)
        if data.get('type') is None or data.get('type') == '':
            data['type'] = 'TextMemoryItem'
            logger.info(f"ParseStoredMemory.from_dict - Missing type field, defaulting to 'TextMemoryItem'")

        # SCORE ALIASING: Map legacy relevance_score to predicted_importance
        # In MongoDB/Parse, this field was called relevance_score but we renamed it to predicted_importance
        # for clearer semantics. This ensures existing stored values are used.
        if data.get('predicted_importance') is None and data.get('relevance_score') is not None:
            data['predicted_importance'] = data['relevance_score']
            logger.debug(f"ParseStoredMemory.from_dict - Aliased relevance_score ({data['relevance_score']}) to predicted_importance")

        logger.info(f"ParseStoredMemory.from_dict - Final data after conversion: {data}")
        instance = cls(**data)
        logger.info(f"ParseStoredMemory.from_dict - Created instance with memoryChunkIds: {instance.memoryChunkIds}")
        return instance

    @field_validator('memoryChunkIds', mode='before')
    @classmethod
    def validate_memory_chunk_ids(cls, v: Any) -> List[str]:
        """Ensure memoryChunkIds is always a list of clean strings"""
        if v is None:
            return []
        
        # If it's already a list, clean each item
        if isinstance(v, list):
            # Clean each item by removing quotes and brackets
            return [str(x).strip().strip("'[]\"") for x in v if x]
            
        # If it's a string that looks like a list
        if isinstance(v, str):
            try:
                # Try to parse as JSON
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(x).strip().strip("'[]\"") for x in parsed if x]
            except json.JSONDecodeError:
                # If it's a string representation of a list
                if v.startswith('[') and v.endswith(']'):
                    # Remove outer brackets and split
                    items = v[1:-1].split(',')
                    return [x.strip().strip("'[]\"") for x in items if x.strip()]
                # If it's a comma-separated string
                if ',' in v:
                    return [x.strip().strip("'[]\"") for x in v.split(',') if x.strip()]
                # Single value
                if v.strip():
                    return [v.strip().strip("'[]\"")]
        
        return []
    
    @classmethod
    def from_parse_response(cls, response_data: Dict[str, Any]) -> 'ParseStoredMemory':
        """Create ParseStoredMemory from Parse Server response"""
        # Extract base fields
        # Resolve org/namespace IDs with pointer fallbacks (legacy fields may be None)
        def _normalize_pointer_id(value):
            if value is None:
                return None
            if isinstance(value, str):
                return value or None
            if isinstance(value, dict):
                if not value:
                    return None
                return value.get('objectId') or value.get('id')
            return getattr(value, 'objectId', None) or getattr(value, 'id', None)

        organization_id = _normalize_pointer_id(response_data.get('organization_id'))
        if not organization_id:
            organization_id = _normalize_pointer_id(response_data.get('organization'))
        if not organization_id:
            organization_id = _normalize_pointer_id(
                (response_data.get('customMetadata') or {}).get('organization_id')
            )
        if not organization_id:
            organization_id = _normalize_pointer_id(
                (response_data.get('metadata') or {}).get('organization_id')
            )

        namespace_id = _normalize_pointer_id(response_data.get('namespace_id'))
        if not namespace_id:
            namespace_id = _normalize_pointer_id(response_data.get('namespace'))
        if not namespace_id:
            namespace_id = _normalize_pointer_id(
                (response_data.get('customMetadata') or {}).get('namespace_id')
            )
        if not namespace_id:
            namespace_id = _normalize_pointer_id(
                (response_data.get('metadata') or {}).get('namespace_id')
            )

        base_data = {
            'objectId': response_data['objectId'],
            'createdAt': response_data['createdAt'],
            'updatedAt': response_data.get('updatedAt'),
            'ACL': response_data.get('ACL', {}),
            'content': response_data['content'],
            'metadata': response_data.get('metadata') or {},  # Ensure not None
            'customMetadata': response_data.get('customMetadata'), 
            'sourceType': response_data.get('sourceType', 'papr'),
            'context': parse_context(response_data.get('context')),
            'title': response_data.get('title'),
            'location': response_data.get('location'),
            'type': response_data.get('type', 'TextMemoryItem'),
            'topics': response_data.get('topics', []),\
            'memoryId': response_data.get('memoryId'),
            'memoryChunkIds': response_data.get('memoryChunkIds', []),
            'user': response_data.get('user'),
            'developerUser': response_data.get('developerUser'),
            'organization': response_data.get('organization'),
            'namespace': response_data.get('namespace'),
            'organization_id': organization_id,
            'namespace_id': namespace_id,
            # Always extract ACL fields from top-level only
            'external_user_read_access': response_data.get('external_user_read_access', []) or [],
            'external_user_write_access': response_data.get('external_user_write_access', []) or [],
            'user_read_access': response_data.get('user_read_access', []) or [],
            'user_write_access': response_data.get('user_write_access', []) or [],
            'workspace_read_access': response_data.get('workspace_read_access', []) or [],
            'workspace_write_access': response_data.get('workspace_write_access', []) or [],
            'role_read_access': response_data.get('role_read_access', []) or [],
            'role_write_access': response_data.get('role_write_access', []) or [],
            'namespace_read_access': response_data.get('namespace_read_access', []) or [],
            'namespace_write_access': response_data.get('namespace_write_access', []) or [],
            'organization_read_access': response_data.get('organization_read_access', []) or [],
            'organization_write_access': response_data.get('organization_write_access', []) or [],
        }

        # Add document-specific fields if this is a DocumentMemoryItem
        if base_data['type'] == "DocumentMemoryItem":
            doc_fields = {
                'page_number': response_data.get('page_number'),
                'total_pages': response_data.get('total_pages'),
                'upload_id': response_data.get('upload_id'),
                'extract_mode': response_data.get('extract_mode'),
                'file_url': response_data.get('file_url'),
                'filename': response_data.get('filename'),
                'page': response_data.get('page')
            }
            base_data.update(doc_fields)

        return cls(**base_data)

class Memory(BaseModel):
    """A memory item in the knowledge base"""
    id: str
    content: str
    title: Optional[str] = None
    type: str
    metadata: Optional[Union[str, Dict[str, Any]]] = None
    external_user_id: Optional[str] = None
    customMetadata: Optional[Dict[str, Any]] = None
    source_type: str = "papr"
    context: Optional[List[ContextItem]] = Field(default_factory=list)
    location: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    hierarchical_structures: str = ""
    source_url: str = ""
    conversation_id: str = ""
    topics: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    current_step: Optional[str] = None
    # Role and category fields for API responses
    role: Optional[str] = Field(default=None, description="Role that generated this memory (user or assistant)")
    category: Optional[str] = Field(default=None, description="Memory category based on role")
    created_at: Optional[datetime] = Field(default=None, serialization_alias="createdAt")
    updated_at: Optional[datetime] = Field(default=None, serialization_alias="updatedAt")
    
    # Access control and ownership
    acl: Dict[str, Dict[str, bool]]
    user_id: str  # ID of the user who owns this memory
    workspace_id: Optional[str] = None  # ID of the workspace this memory belongs to
    
    # Multi-tenant fields (NEW) - kept as string IDs in public API for simplicity
    organization_id: Optional[str] = Field(default=None, description="Organization ID that owns this memory")
    namespace_id: Optional[str] = Field(default=None, description="Namespace ID this memory belongs to")
    
    # Source context
    source_document_id: Optional[str] = None  # Previously 'post' - ID of the document/page this memory is from
    source_message_id: Optional[str] = None  # Previously 'postMessage' - ID of the chat message this memory is from
    
    # Document specific fields
    page_number: Optional[int] = None
    total_pages: Optional[int] = None
    file_url: Optional[str] = None
    filename: Optional[str] = None
    page: Optional[str] = None

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

    # Embedding fields (optional, populated when include_embeddings=True in sync_tiers)
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Full precision (float32) embedding vector from Qdrant. Typically 2560 dimensions for Qwen4B. Used for CoreML/ANE fp16 models."
    )
    embedding_int8: Optional[List[int]] = Field(
        default=None,
        description="Quantized INT8 embedding vector (values -128 to 127). 4x smaller than float32. Default format for efficiency."
    )

    # =============================================================================
    # RELEVANCE SCORES - Research-backed scoring system
    # =============================================================================
    # References:
    #   - BM25/IR: log1p() normalization for frequency (prevents popularity bias)
    #   - RRF (Cormack et al. 2009): score = Σ 1/(k + rank_i) for rank fusion
    #   - Time decay: exp(-λ * age) with configurable half-life
    #   - Multi-signal fusion outperforms single signals (LTR research)
    # =============================================================================

    # COMPUTED AT SEARCH TIME - Base signals:
    similarity_score: Optional[float] = Field(
        default=None,
        description="Cosine similarity from vector search (0-1). Measures semantic relevance to query."
    )
    popularity_score: Optional[float] = Field(
        default=None,
        description="Popularity signal (0-1): 0.5*cacheConfidenceWeighted30d + 0.5*citationConfidenceWeighted30d. Uses stored EMA fields."
    )
    recency_score: Optional[float] = Field(
        default=None,
        description="Recency signal (0-1): exp(-0.05 * days_since_last_access). Half-life ~14 days."
    )

    # ONLY when rank_results=True - Reranker signals:
    reranker_score: Optional[float] = Field(
        default=None,
        description="Reranker relevance (0-1). From cross-encoder (Cohere/Qwen3/BGE) or LLM (GPT-5-nano)."
    )
    reranker_confidence: Optional[float] = Field(
        default=None,
        description="Reranker confidence (0-1). Meaningful for LLM reranking; equals reranker_score for cross-encoders."
    )
    reranker_type: Optional[str] = Field(
        default=None,
        description="Reranker type: 'cross_encoder' (Cohere/Qwen3/BGE) or 'llm' (GPT-5-nano/GPT-4o-mini)."
    )

    # FINAL COMBINED SCORE:
    relevance_score: Optional[float] = Field(
        default=None,
        description="Final relevance (0-1). rank_results=False: 0.6*sim + 0.25*pop + 0.15*recency. rank_results=True: RRF-based fusion."
    )

    # Processing metrics (optional; populated when available from Parse)
    metrics: Optional[Dict[str, Any]] = None
    totalProcessingCost: Optional[float] = None

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='allow'
    )
    
    # Note: Removed serialize_relevance_score fallback - scores should be None if not computed
    # The new scoring fields (predicted_importance, behavioral_score, etc.) provide clear semantics

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

    @classmethod
    def from_internal(cls, parse_memory: ParseStoredMemory) -> 'Memory':
        """Convert internal ParseStoredMemory to public-facing Memory"""
        user_id = None
        # 1. Try user_read_access
        if getattr(parse_memory, 'user_read_access', None) and len(parse_memory.user_read_access) > 0:
            user_id = parse_memory.user_read_access[0]
        # 2. Try user pointer (dict or object)
        elif getattr(parse_memory, 'user', None):
            user_obj = parse_memory.user
            if isinstance(user_obj, dict) and 'objectId' in user_obj:
                user_id = user_obj['objectId']
            elif hasattr(user_obj, 'objectId'):
                user_id = getattr(user_obj, 'objectId')
        # 3. Try ACL keys
        elif getattr(parse_memory, 'ACL', None):
            acl_keys = list(parse_memory.ACL.keys())
            if acl_keys:
                user_id = acl_keys[0]
        # 4. Fallback to empty string
        if user_id is None:
            user_id = ""
        # Robust extraction of external_user_id
        external_user_id = None
        if getattr(parse_memory, 'developerUser', None):
            dev_user = parse_memory.developerUser
            if isinstance(dev_user, dict):
                external_user_id = dev_user.get('external_id')
            else:
                external_user_id = getattr(dev_user, 'external_id', None)
        if not external_user_id:
            external_user_id = getattr(parse_memory, 'external_user_id', None)
        if not external_user_id:
            # Fallback: if only one external user in ACL, use it
            acl_list = getattr(parse_memory, 'external_user_read_access', [])
            if isinstance(acl_list, list) and len(acl_list) == 1:
                external_user_id = acl_list[0]
        # Resolve org/namespace IDs with pointer fallbacks (legacy fields may be None)
        def _normalize_pointer_id(value):
            if value is None:
                return None
            if isinstance(value, str):
                return value or None
            if isinstance(value, dict):
                if not value:
                    return None
                return value.get('objectId') or value.get('id')
            return getattr(value, 'objectId', None) or getattr(value, 'id', None)

        organization_id = _normalize_pointer_id(getattr(parse_memory, 'organization_id', None))
        if not organization_id and getattr(parse_memory, 'organization', None):
            organization_id = _normalize_pointer_id(parse_memory.organization)
        if not organization_id:
            organization_id = _normalize_pointer_id(
                getattr(parse_memory, 'customMetadata', None) and (parse_memory.customMetadata or {}).get('organization_id')
            )
        if not organization_id:
            organization_id = _normalize_pointer_id(
                getattr(parse_memory, 'metadata', None) and (parse_memory.metadata or {}).get('organization_id')
            )

        namespace_id = _normalize_pointer_id(getattr(parse_memory, 'namespace_id', None))
        if not namespace_id and getattr(parse_memory, 'namespace', None):
            namespace_id = _normalize_pointer_id(parse_memory.namespace)
        if not namespace_id:
            namespace_id = _normalize_pointer_id(
                getattr(parse_memory, 'customMetadata', None) and (parse_memory.customMetadata or {}).get('namespace_id')
            )
        if not namespace_id:
            namespace_id = _normalize_pointer_id(
                getattr(parse_memory, 'metadata', None) and (parse_memory.metadata or {}).get('namespace_id')
            )
        # Normalize ACL keys for API response (r/w -> read/write)
        acl = parse_memory.ACL or {}
        if isinstance(acl, dict):
            normalized_acl = {}
            for principal, perms in acl.items():
                if isinstance(perms, dict):
                    normalized_perms = dict(perms)
                    if 'r' in normalized_perms and 'read' not in normalized_perms:
                        normalized_perms['read'] = normalized_perms['r']
                    if 'w' in normalized_perms and 'write' not in normalized_perms:
                        normalized_perms['write'] = normalized_perms['w']
                    normalized_acl[principal] = normalized_perms
                else:
                    normalized_acl[principal] = perms
            acl = normalized_acl
        base_data = {
            'id': parse_memory.memoryId,
            'content': parse_memory.content,
            'title': parse_memory.title,
            'type': parse_memory.type,
            'metadata': parse_memory.metadata,
            'external_user_id': external_user_id,
            'customMetadata': getattr(parse_memory, 'customMetadata', None),
            'source_type': getattr(parse_memory, 'sourceType', 'papr'),
            'context': getattr(parse_memory, 'context', []),
            'location': getattr(parse_memory, 'location', None),
            'tags': getattr(parse_memory, 'emojiTags', []),
            'hierarchical_structures': getattr(parse_memory, 'hierarchicalStructures', ''),
            'source_url': getattr(parse_memory, 'sourceUrl', ''),
            'conversation_id': getattr(parse_memory, 'conversationId', ''),
            'topics': getattr(parse_memory, 'topics', []),
            'steps': getattr(parse_memory, 'steps', []),
            'current_step': getattr(parse_memory, 'current_step', None),
            'user_id': user_id,
            'created_at': parse_memory.createdAt,
            'updated_at': parse_memory.updatedAt,
            # Access control and ownership
            'acl': acl,
            'workspace_id': parse_memory.workspace.objectId if parse_memory.workspace else None,
            # Multi-tenant fields (NEW)
            'organization_id': organization_id,
            'namespace_id': namespace_id,
            # Source context with renamed fields
            'source_document_id': parse_memory.post.objectId if parse_memory.post else None,
            'source_message_id': parse_memory.postMessage.objectId if parse_memory.postMessage else None,
            # ACL fields from top-level ParseStoredMemory fields
            'external_user_read_access': getattr(parse_memory, 'external_user_read_access', []) or [],
            'external_user_write_access': getattr(parse_memory, 'external_user_write_access', []) or [],
            'user_read_access': getattr(parse_memory, 'user_read_access', []) or [],
            'user_write_access': getattr(parse_memory, 'user_write_access', []) or [],
            'workspace_read_access': getattr(parse_memory, 'workspace_read_access', []) or [],
            'workspace_write_access': getattr(parse_memory, 'workspace_write_access', []) or [],
            'role_read_access': getattr(parse_memory, 'role_read_access', []) or [],
            'role_write_access': getattr(parse_memory, 'role_write_access', []) or [],
            'namespace_read_access': getattr(parse_memory, 'namespace_read_access', []) or [],
            'namespace_write_access': getattr(parse_memory, 'namespace_write_access', []) or [],
            'organization_read_access': getattr(parse_memory, 'organization_read_access', []) or [],
            'organization_write_access': getattr(parse_memory, 'organization_write_access', []) or [],
            # Role and category fields
            'role': getattr(parse_memory, 'role', None),
            'category': getattr(parse_memory, 'category', None),
            # Relevance scores - Research-backed scoring system
            'similarity_score': getattr(parse_memory, 'similarity_score', None),
            'popularity_score': getattr(parse_memory, 'popularity_score', None),
            'recency_score': getattr(parse_memory, 'recency_score', None),
            'reranker_score': getattr(parse_memory, 'reranker_score', None),
            'reranker_confidence': getattr(parse_memory, 'reranker_confidence', None),
            'reranker_type': getattr(parse_memory, 'reranker_type', None),
            'relevance_score': getattr(parse_memory, 'relevance_score', None),
            # Processing metrics (if present in Parse)
            'metrics': getattr(parse_memory, 'metrics', None),
            'totalProcessingCost': (
                getattr(parse_memory, 'totalProcessingCost', None)
                or getattr(parse_memory, 'total_processing_cost', None)
            ),
        }

        # Add document-specific fields if this is a DocumentMemoryItem
        if parse_memory.type == "DocumentMemoryItem":
            doc_fields = {
                'page_number': parse_memory.page_number,
                'total_pages': parse_memory.total_pages,
                'file_url': parse_memory.file_url,
                'filename': parse_memory.filename,
                'page': parse_memory.page
            }
            base_data.update(doc_fields)

        return cls(**base_data)

    def without_metadata(self) -> 'Memory':
        """Create a copy of the memory item without metadata field."""
        return Memory.model_validate(
            self.model_dump(exclude={'metadata'})
        )
    
class MemoryRetrievalResult(TypedDict):
    results: List[ParseStoredMemory]
    missing_memory_ids: List[str]


class UseCaseMetrics(TypedDict):
    usecase_token_count_input: int
    usecase_token_count_output: int
    usecase_total_cost: float

class UseCaseResponse(TypedDict):
    data: Dict[str, List[Dict[str, Any]]]  # Contains 'goals' and 'use_cases' lists
    metrics: UseCaseMetrics
    refusal: Optional[str]
   
class RelatedMemoriesMetrics(TypedDict):
    related_memories_token_count_input: int
    related_memories_token_count_output: int
    related_memories_total_cost: float

class MemoryWithConfidence(TypedDict):
    memory: ParseStoredMemory
    confidence_score: float

class RelatedMemoriesSuccess(TypedDict):
    data: List[ParseStoredMemory]
    generated_queries: List[str]
    confidence_scores: Optional[List[float]]  # Parallel to data list, optional since ranking isn't always performed
    metrics: RelatedMemoriesMetrics

class RelatedMemoriesError(TypedDict):
    error: str
    generated_queries: List[str]
    confidence_scores: Optional[List[float]]  # Parallel to generated_queries list, optional since ranking isn't always performed
    metrics: RelatedMemoriesMetrics

class DeletionStatus(BaseModel):
    pinecone: bool = False
    neo4j: bool = False
    parse: bool = False
    qdrant: bool = False

class DeleteMemoryResponse(BaseModel):
    code: int = Field(default=200, description="HTTP status code")
    status: str = Field(default="success", description="'success' or 'error'")
    message: Optional[str] = None
    error: Optional[str] = None
    memoryId: str = ""
    objectId: str = ""
    deletion_status: Optional[DeletionStatus] = None  # Rename to avoid confusion with status str
    details: Optional[Any] = None  # For extra error context

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

    @classmethod
    def success(cls, memoryId: str, objectId: str, deletion_status: DeletionStatus, code: int = 200, message: Optional[str] = None):
        return cls(
            code=code,
            status="success",
            message=message,
            error=None,
            memoryId=memoryId,
            objectId=objectId,
            deletion_status=deletion_status,
            details=None
        )

    @classmethod
    def failure(cls, error: str, code: int = 400, message: Optional[str] = None, details: Any = None):
        return cls(
            code=code,
            status="error",
            message=message,
            error=error,
            memoryId="",
            objectId="",
            deletion_status=None,
            details=details
        )

class DeleteMemoryResult(TypedDict):
    response: DeleteMemoryResponse
    status_code: int

# Define types for Pinecone response
class PineconeMatch(TypedDict):
    id: str
    score: float
    metadata: dict
    values: Optional[List[float]]

class PineconeQueryResponse(TypedDict):
    matches: List[PineconeMatch]
    namespace: str

# Error response model
class ErrorResponse(BaseModel):
    error: str

class InteractionLimits(TypedDict):
    mini: int
    premium: int

class TierLimits(TypedDict):
    pro: InteractionLimits
    business_plus: InteractionLimits
    free_trial: InteractionLimits
    enterprise: InteractionLimits

class SystemUpdateStatus(BaseModel):
    """Status of update operation for each system"""
    pinecone: bool = False
    neo4j: bool = False
    parse: bool = False

class UpdateMemoryItem(BaseModel):
    """Model for a single updated memory item"""
    objectId: str
    memoryId: str
    content: Optional[str] = "" 
    updatedAt: datetime
    memoryChunkIds: Optional[List[str]] = [],
    metadata: Optional[MemoryMetadata] = None

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

    @field_serializer('updatedAt', when_used='json')
    def serialize_updated_at(self, v):
        return v.isoformat() if v else None

    def model_dump(self, *args, **kwargs):
        d = super().model_dump(*args, **kwargs)
        if isinstance(d.get('updatedAt'), datetime):
            d['updatedAt'] = d['updatedAt'].isoformat()
        return d

class UpdateMemoryResponse(BaseModel):
    """Unified response model for update_memory API endpoint (success or error)."""
    code: int = Field(default=200, description="HTTP status code")
    status: str = Field(default="success", description="'success' or 'error'")
    memory_items: Optional[List[UpdateMemoryItem]] = Field(default=None, description="List of updated memory items if successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    details: Optional[Any] = Field(default=None, description="Additional error details or context")
    message: Optional[str] = Field(default=None, description="Status message")
    status_obj: Optional[SystemUpdateStatus] = Field(default=None, description="System update status (pinecone, neo4j, parse)")

    @field_serializer('memory_items', when_used='json')
    def serialize_memory_items(self, memory_items):
        # If memory_items is a list of dicts or models, ensure any 'updatedAt' is ISO string
        if memory_items:
            for item in memory_items:
                if isinstance(item, dict):
                    if isinstance(item.get('updatedAt'), datetime):
                        item['updatedAt'] = item['updatedAt'].isoformat()
                else:
                    # It's a Pydantic model
                    if hasattr(item, 'updatedAt') and isinstance(item.updatedAt, datetime):
                        item.updatedAt = item.updatedAt.isoformat()
        return memory_items

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid',
    )

    @classmethod
    def success(cls, memory_items: List[UpdateMemoryItem], status_obj: SystemUpdateStatus, code: int = 200, message: str = "Memory item(s) updated"):
        return cls(
            code=code,
            status="success",
            memory_items=memory_items,
            error=None,
            details=None,
            message=message,
            status_obj=status_obj
        )

    @classmethod
    def failure(cls, error: str, code: int = 400, details: Any = None, message: str = "Update failed", status_obj: Optional[SystemUpdateStatus] = None):
        return cls(
            code=code,
            status="error",
            memory_items=None,
            error=error,
            details=details,
            message=message,
            status_obj=status_obj
        )

class MemoryParseServer(BaseModel):
    """Model that exactly matches the Parse Server Memory class schema"""
    objectId: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    ACL: Dict[str, Dict[str, bool]]
    title: Optional[str] = None
    content: Optional[str] = None
    file: Optional[Dict[str, Any]] = None
    user: ParsePointer
    workspace: Optional[ParsePointer] = None
    topics: Optional[List[str]] = Field(default_factory=list)
    location: Optional[str] = None
    emojiTags: Optional[List[str]] = Field(default_factory=list)
    emotionTags: Optional[List[str]] = Field(default_factory=list)
    hierarchicalStructures: Optional[str] = None
    type: Optional[str] = None
    sourceUrl: Optional[str] = None
    conversationId: Optional[str] = None
    # Role and category as primary fields in Parse Server
    role: Optional[str] = Field(default=None, description="Role that generated this memory (user or assistant)")
    category: Optional[str] = Field(default=None, description="Memory category based on role")
    memoryId: Optional[str] = None
    imageURL: Optional[str] = None
    sourceType: Optional[str] = Field(default="papr")
    context: Optional[List[ContextItem]] = Field(default_factory=list)
    goals: Optional[List[Dict[str, Any]]] = None   
    usecases: Optional[List[Dict[str, Any]]] = None   
    node: Optional[ParsePointer] = None
    relationship_json: Optional[List[Any]] = Field(default_factory=list)
    node_name: Optional[str] = None
    postMessage: Optional[ParsePointer] = None
    current_step: Optional[str] = None
    steps: Optional[List[str]] = Field(default_factory=list)
    post: Optional[ParsePointer] = None
    totalCost: Optional[float] = None
    tokenSize: Optional[int] = None
    storageSize: Optional[int] = None
    usecaseGenerationCost: Optional[float] = None
    schemaGenerationCost: Optional[float] = None
    relatedMemoriesCost: Optional[float] = None
    nodeDefinitionCost: Optional[float] = None
    bigbirdEmbeddingCost: Optional[float] = None
    sentenceBertCost: Optional[float] = None
    totalProcessingCost: Optional[float] = None
    memoryIds: Optional[List[str]] = Field(default_factory=list)
    memoryChunkIds: Optional[List[str]] = Field(default_factory=list)
    metrics: Optional[Dict[str, Any]] = Field(default_factory=dict)

    # New DocumentMemoryItem specific fields
    page_number: Optional[int] = None
    total_pages: Optional[int] = None
    upload_id: Optional[str] = None
    extract_mode: Optional[str] = None
    file_url: Optional[str] = None  # Parse Server file URL
    filename: Optional[str] = None
    page: Optional[str] = None  # Format: "X of Y"

    developerUser: Optional[ParsePointer] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    customMetadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # Multi-tenant fields (NEW) with Parse Pointers
    organization: Optional[OrganizationPointer] = Field(default=None, description="Organization (Pointer) that owns this memory")
    namespace: Optional[NamespacePointer] = Field(default=None, description="Namespace (Pointer) this memory belongs to")
    
    # Deprecated (kept for backward compatibility)
    organization_id: Optional[str] = Field(default=None, description="DEPRECATED: Use organization pointer instead")
    namespace_id: Optional[str] = Field(default=None, description="DEPRECATED: Use namespace pointer instead")

    # Counters and EMA fields for cache and citation usage
    cacheHitTotal: Optional[int] = None
    cacheHitEma30d: Optional[float] = None
    cacheConfidenceWeighted30d: Optional[float] = None
    cacheEmaUpdatedAt: Optional[datetime] = None
    citationHitTotal: Optional[int] = None
    citationHitEma30d: Optional[float] = None
    citationConfidenceWeighted30d: Optional[float] = None
    citationEmaUpdatedAt: Optional[datetime] = None
    lastAccessedAt: Optional[datetime] = None

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

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='allow',
    )

    def model_dump(self, *args, **kwargs):
        """Override model_dump to ensure proper handling of ACL, pointer fields, and DocumentMemoryItem"""
        data = super().model_dump(*args, **kwargs)
        
        # Helper function to transform pointer fields
        def transform_pointer(pointer_data):
            if pointer_data and isinstance(pointer_data, dict):
                if 'type' in pointer_data:
                    pointer_data['__type'] = pointer_data.pop('type')
            return pointer_data
        
        # Transform all pointer fields
        pointer_fields = ['user', 'workspace', 'node', 'postMessage', 'post']
        for field in pointer_fields:
            if field in data:
                data[field] = transform_pointer(data[field])
        
        # Ensure ACL is properly formatted
        if 'ACL' in data and isinstance(data['ACL'], dict):
            # If ACL is a dict of single characters, reconstruct it properly
            if any(len(k) == 1 for k in data['ACL'].keys()):
                reconstructed_acl = {}
                current_key = []
                
                for k, v in sorted(data['ACL'].items()):
                    if k.startswith('role:'):
                        reconstructed_acl[k] = v
                    else:
                        current_key.append(k)
                        if k in ['[', ']']:
                            key = ''.join(current_key)
                            if key not in reconstructed_acl:
                                reconstructed_acl[key] = v
                            current_key = []
                
                data['ACL'] = reconstructed_acl
        
        # Handle DocumentMemoryItem fields
        if data.get('type') != 'DocumentMemoryItem':
            for field in ['extract_mode']:
                data.pop(field, None)
        
        return data

    @field_validator('memoryIds', 'topics', 'emojiTags', 'steps', 'relationship_json')
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            try:
                # Try to parse if it's a JSON string
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                # If it's a comma-separated string or string representation of a list
                if v.startswith('[') and v.endswith(']'):
                    # Handle string representation of a list
                    try:
                        # Remove brackets and split by comma
                        items = v[1:-1].split(',')
                        # Clean up each item
                        return [item.strip().strip("'\"") for item in items if item.strip()]
                    except Exception:
                        return []
                return [x.strip() for x in v.split(',') if x.strip()]
        if isinstance(v, list):
            return v
        return []

    @field_validator('memoryChunkIds')
    @classmethod
    def validate_memory_chunk_ids(cls, v: Any) -> List[str]:
        """Ensure memoryChunkIds is always a list of strings"""
        if v is None:
            return []
        
        # If it's already a list, clean it up
        if isinstance(v, list):
            return [str(x).strip() for x in v if x]
            
        # If it's a string that looks like a list
        if isinstance(v, str):
            try:
                # Try to parse as JSON
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if x]
            except json.JSONDecodeError:
                # If it's a comma-separated string
                if ',' in v:
                    return [x.strip() for x in v.split(',') if x.strip()]
                # Single value
                if v.strip():
                    return [v.strip()]
        
        return []

    @field_validator('steps')
    @classmethod
    def format_steps(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            try:
                # Try to parse if it's a JSON string
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    # Clean up any string representations of lists
                    return [item.strip("[]'\" ") for item in parsed]
                return []
            except json.JSONDecodeError:
                # If it's a string representation of a list
                if v.startswith('[') and v.endswith(']'):
                    try:
                        # Remove brackets and split by comma
                        items = v[1:-1].split(',')
                        # Clean up each item
                        return [item.strip().strip("[]'\" ") for item in items if item.strip()]
                    except Exception:
                        return []
                # If it's a comma-separated string
                return [step.strip().strip("[]'\" ") for step in v.split(',') if step.strip()]
        if isinstance(v, list):
            # Clean up any string representations of lists in the array
            return [item.strip("[]'\" ") if isinstance(item, str) else item for item in v]
        return []

    @field_validator('emojiTags')
    @classmethod
    def format_emoji_tags(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            # Handle comma-separated string
            return [tag.strip() for tag in v.split(',') if tag.strip()]
        if isinstance(v, list):
            return v
        return []

    @field_validator('emotionTags')
    @classmethod
    def format_emotion_tags(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            # Handle comma-separated string
            return [tag.strip() for tag in v.split(',') if tag.strip()]
        if isinstance(v, list):
            return v
        return []

    @field_validator('topics')
    @classmethod
    def format_topics(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            # Handle comma-separated string
            return [topic.strip() for topic in v.split(',') if topic.strip()]
        if isinstance(v, list):
            return v
        return []

    @field_validator('hierarchicalStructures')
    @classmethod
    def format_hierarchical_structures(cls, v):
        if v is None:
            return ""
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x) for x in v)
        if isinstance(v, dict):
            return json.dumps(v)
        return str(v)

    @field_validator('*', mode='before')
    @classmethod
    def handle_none(cls, v):
        return v if v is not None else None

    @model_validator(mode="before")
    @classmethod
    def fix_parse_relations(cls, data):
        # Ensure relatedGoals, relatedUseCases, relatedSteps are lists, not relation dicts
        for field in ["relatedGoals", "relatedUseCases", "relatedSteps"]:
            v = data.get(field)
            if isinstance(v, dict) and v.get("__type") == "Relation":
                data[field] = []
        return data


class MemoryParseServerUpdate(BaseModel):
    """Model for updating existing Memory items in Parse Server"""
    ACL: Optional[Dict[str, Dict[str, bool]]] = None
    content: Optional[str] = None
    sourceType: Optional[str] = Field(default="papr")
    context: Optional[List[ContextItem]] = Field(default_factory=list)
    title: Optional[str] = None
    location: Optional[str] = None
    emojiTags: Optional[List[str]] = None
    emotionTags: Optional[List[str]] = None
    hierarchicalStructures: Optional[str] = None
    type: Optional[str] = None
    sourceUrl: Optional[str] = None
    conversationId: Optional[str] = None
    # Role and category as primary fields in Parse Server
    role: Optional[str] = Field(default=None, description="Role that generated this memory (user or assistant)")
    category: Optional[str] = Field(default=None, description="Memory category based on role")
    topics: Optional[List[str]] = None
    steps: Optional[List[str]] = None
    current_step: Optional[str] = None
    memoryChunkIds: Optional[List[str]] = Field(default_factory=list)
    relationship_json: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    memoryIds: Optional[List[str]] = Field(default_factory=list)
    totalCost: Optional[float] = None
    tokenSize: Optional[int] = None
    storageSize: Optional[int] = None
    usecaseGenerationCost: Optional[float] = None
    schemaGenerationCost: Optional[float] = None
    relatedMemoriesCost: Optional[float] = None
    nodeDefinitionCost: Optional[float] = None
    bigbirdEmbeddingCost: Optional[float] = None
    sentenceBertCost: Optional[float] = None
    totalProcessingCost: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = Field(default_factory=dict)
    page_number: Optional[int] = None
    total_pages: Optional[int] = None
    upload_id: Optional[str] = None
    filename: Optional[str] = None
    page: Optional[str] = None
    developerUser: Optional[ParsePointer] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    customMetadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # Multi-tenant fields (NEW) with Parse Pointers
    organization: Optional[OrganizationPointer] = Field(default=None, description="Organization (Pointer) that owns this memory")
    namespace: Optional[NamespacePointer] = Field(default=None, description="Namespace (Pointer) this memory belongs to")
    
    # Deprecated (kept for backward compatibility)
    organization_id: Optional[str] = Field(default=None, description="DEPRECATED: Use organization pointer instead")
    namespace_id: Optional[str] = Field(default=None, description="DEPRECATED: Use namespace pointer instead")

    # Optional counters and EMA update fields (updates only)
    cacheHitTotal: Optional[int] = None
    cacheHitEma30d: Optional[float] = None
    cacheConfidenceWeighted30d: Optional[float] = None
    cacheEmaUpdatedAt: Optional[datetime] = None
    citationHitTotal: Optional[int] = None
    citationHitEma30d: Optional[float] = None
    citationConfidenceWeighted30d: Optional[float] = None
    citationEmaUpdatedAt: Optional[datetime] = None
    lastAccessedAt: Optional[datetime] = None

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

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='allow',
    )

    @field_validator('memoryChunkIds')
    @classmethod
    def validate_memory_chunk_ids(cls, v: Optional[Any]) -> List[str]:
        """Ensure memoryChunkIds is a list of strings"""
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x]
        return []

    @field_validator('emojiTags', 'emotionTags', 'topics', 'steps')
    @classmethod
    def ensure_list(cls, v: Optional[Any]) -> List[str]:
        """Ensure fields are always lists"""
        if v is None:
            return []
        if isinstance(v, str):
            return [x.strip() for x in v.split(',') if x.strip()]
        if isinstance(v, list):
            return [str(x).strip() for x in v if x]
        return []

    @field_validator('hierarchicalStructures')
    @classmethod
    def format_hierarchical_structures(cls, v: Optional[Any]) -> str:
        """Format hierarchical structures as string"""
        if v is None:
            return ""
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x) for x in v)
        if isinstance(v, dict):
            return json.dumps(v)
        return str(v)
    

class DocumentUploadStatusType(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    NOT_FOUND = "not_found"
    QUEUED = "queued"  # Optional: if we want to indicate queued state
    CANCELLED = "cancelled"  # Optional: if we want to support cancellation


class DocumentUploadStatus(BaseModel):
    progress: float = Field(..., description="0.0 to 1.0 for percentage")
    current_page: Optional[int] = None
    total_pages: Optional[int] = None
    current_filename: Optional[str] = None
    upload_id: Optional[str] = None
    page_id: Optional[str] = Field(default=None, description="Post ID in Parse Server (user-facing page ID)")
    status_type: Optional[DocumentUploadStatusType] = Field(default=None, description="Processing status type")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

class DocumentUploadResponse(BaseModel):
    code: int = Field(default=200, description="HTTP status code")
    status: str = Field(default="success", description="'success', 'processing', 'error', etc.")
    message: Optional[str] = Field(default=None, description="Human-readable status message")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    details: Optional[Any] = Field(default=None, description="Additional error details or context")
    document_status: DocumentUploadStatus = Field(..., description="Status and progress of the document upload")
    memory_items: List[AddMemoryItem] = Field(default_factory=list, description="List of memory items created from the document")
    memories: Optional[List[AddMemoryItem]] = Field(default=None, description="For backward compatibility")

    @field_validator('memories', mode='before')
    @classmethod
    def set_memories_from_memory_items(cls, v, info):
        """Ensure memories field matches memory_items for backward compatibility"""
        if v is None and info.data.get('memory_items'):
            return info.data['memory_items']
        return v

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

    @classmethod
    def success(cls, document_status: DocumentUploadStatus, memory_items: Optional[List[AddMemoryItem]] = None, code: int = 200, message: Optional[str] = None, details: Any = None):
        return cls(
            code=code,
            status="success",
            message=message,
            error=None,
            details=details,
            document_status=document_status,
            memory_items=memory_items or [],
            memories=memory_items or []
        )

    @classmethod
    def failure(cls, error: str, document_status: Optional[DocumentUploadStatus] = None, code: int = 400, message: Optional[str] = None, details: Any = None, memory_items: Optional[List[AddMemoryItem]] = None):
        return cls(
            code=code,
            status="error",
            message=message,
            error=error,
            details=details,
            document_status=document_status or DocumentUploadStatus(progress=0.0, status_type=DocumentUploadStatusType.FAILED, error=error),
            memory_items=memory_items or [],
            memories=memory_items or []
        )

class DocumentUploadStatusResponse(BaseModel):
    """Response model for document upload status from Parse Server"""
    objectId: str
    status: DocumentUploadStatusType
    filename: Optional[str] = None
    progress: float
    current_page: Optional[int] = None
    total_pages: Optional[int] = None
    current_filename: Optional[str] = None
    error: Optional[str] = None
    upload_id: str
    user: ParseUserPointer

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

class ParseFileUploadResponse(BaseModel):
    """Response model for file uploads to Parse Server"""
    file_url: str
    source_url: str
    name: str
    mime_type: str

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra='forbid'
    )

class ParseFile(BaseModel):
    """Pointer-like representation of a Parse File object."""
    name: str
    url: Optional[str] = None
    type: str = Field(default="File", alias="__type")

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='forbid'
    )

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        if 'type' in data:
            data['__type'] = data.pop('type')
        return data

class PostParseServer(BaseModel):
    """Model for Post class in Parse Server."""
    objectId: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    ACL: Dict[str, Dict[str, bool]] = Field(default_factory=dict)
    className: Optional[str] = Field(default="Post")

    # Content and text
    content: Dict[str, Any] = Field(default_factory=dict)
    text: Optional[str] = None  # Markdown/plain text representation

    # Ownership and context
    user: Optional[ParsePointer] = None
    workspace: Optional[ParsePointer] = None

    # File pointer (stored file)
    file: Optional[ParseFile] = None

    # Document processing specific fields
    uploadId: Optional[str] = None
    documentProcessed: Optional[bool] = None
    post_title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    processingMetadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # Provider result storage (for large JSON files)
    provider_result_file: Optional[ParseFile] = None  # Compressed provider JSON stored as file
    providerResultFile: Optional[ParseFile] = None  # camelCase alias for Parse compatibility
    
    # Extraction result storage (for large extraction results)
    extraction_result_file: Optional[ParseFile] = None  # Compressed extraction stored as file
    extractionResultFile: Optional[ParseFile] = None  # camelCase alias for Parse compatibility
    extractionMetadata: Optional[Dict[str, Any]] = Field(default_factory=dict)  # Summary of extraction

    # ========================================================================
    # DOCUMENT PROCESSING PIPELINE TRACKING
    # Each stage stores its output separately for analysis and debugging
    # ========================================================================

    # STAGE 1: Provider Extraction (Reducto/OCR)
    provider_extraction: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Provider (Reducto/OCR) extraction output with stats and URL"
    )
    # Schema: {
    #   url: str,  # URL to compressed extraction file
    #   provider: str,  # "reducto", "tensorlake", etc.
    #   stats: {
    #     total_elements: int,
    #     text: int, table: int, image: int,
    #     extraction_size_bytes: int,
    #     compression_ratio: float
    #   },
    #   timestamp: ISO8601,
    #   duration_ms: int
    # }

    # STAGE 2: Hierarchical Chunking
    chunked_extraction: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Hierarchically chunked extraction with semantic grouping stats"
    )
    # Schema: {
    #   url: str,  # URL to compressed chunked extraction
    #   stats: {
    #     original_count: int,
    #     chunked_count: int,
    #     reduction_percent: float,
    #     element_types: dict,
    #     avg_chunk_size: int,
    #     compression_ratio: float
    #   },
    #   config: {strategy: str, max_chunk_size: int, min_chunk_size: int},
    #   timestamp: ISO8601,
    #   duration_ms: int
    # }

    # STAGE 3: LLM Metadata Enhancement
    llm_enhanced_extraction: Optional[Dict[str, Any]] = Field(
        default=None,
        description="LLM-enhanced extraction with metadata and chunking validation"
    )
    # Schema: {
    #   url: str,  # URL to compressed LLM-enhanced extraction
    #   stats: {
    #     elements_processed: int,
    #     llm_calls: int,
    #     total_tokens_used: int,
    #     avg_tokens_per_chunk: int,
    #     model: str,  # "gpt-5-nano", etc.
    #     failed_elements: int,
    #     chunking_validation_results: {
    #       coherent: int, incomplete: int, mixed_topics: int
    #     }
    #   },
    #   timestamp: ISO8601,
    #   duration_ms: int
    # }

    # STAGE 4: Memory Requests (Ready for Indexing)
    memory_requests_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Final memory requests ready for bulk indexing"
    )
    # Schema: {
    #   url: str,  # URL to compressed memory requests
    #   stats: {
    #     total_requests: int,
    #     estimated_size_bytes: int,
    #     ready_for_indexing: bool
    #   },
    #   timestamp: ISO8601
    # }

    # STAGE 5: Indexing Results
    indexing_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Results of indexing memories to Neo4j/Qdrant/Parse"
    )
    # Schema: {
    #   status: str,  # "completed", "partial", "failed"
    #   stats: {
    #     successful: int,
    #     failed: int,
    #     neo4j_nodes: int,
    #     qdrant_vectors: int,
    #     parse_memories: int
    #   },
    #   timestamp: ISO8601,
    #   duration_ms: int
    # }

    # Overall Pipeline Tracking
    pipeline_status: Optional[str] = Field(
        default=None,
        description="Overall pipeline status: pending, processing, completed, failed"
    )
    pipeline_start: Optional[datetime] = Field(
        default=None,
        description="Pipeline start timestamp"
    )
    pipeline_end: Optional[datetime] = Field(
        default=None,
        description="Pipeline end timestamp"
    )
    total_duration_ms: Optional[int] = Field(
        default=None,
        description="Total pipeline duration in milliseconds"
    )

    # Backward compatibility fields
    extraction_result: Optional[str] = Field(
        default=None,
        description="Quick access to latest extraction URL (for backward compatibility)"
    )
    extraction_stored: Optional[bool] = Field(
        default=False,
        description="Whether extraction is stored in Parse (vs inline)"
    )

    # Source descriptors
    type: Optional[str] = Field(default="page")
    source: Optional[str] = Field(default="/document upload API")
    mediaType: Optional[str] = None
    thumbnailRatio: Optional[float] = None

    # Multi-tenant (string ids for compatibility)
    organizationId: Optional[str] = None
    namespaceId: Optional[str] = None

    # Post status and flags
    hasSignificantUpdate: Optional[bool] = Field(default=False)
    needsMemoryUpdate: Optional[bool] = Field(default=True)
    needsImmediateUpdate: Optional[bool] = Field(default=True)
    isNew: Optional[bool] = Field(default=False)
    archive: Optional[bool] = Field(default=False)
    hasURL: Optional[bool] = None

    # Social and interaction counts
    likesCount: Optional[int] = Field(default=0)
    postSocialCount: Optional[int] = Field(default=0)
    isIncognito: Optional[bool] = Field(default=False)
    questionAnswerEnabled: Optional[bool] = Field(default=True)
    chatEnabled: Optional[bool] = Field(default=True)
    postMessageCount: Optional[int] = Field(default=0)
    postMessageUnReadCount: Optional[int] = Field(default=0)
    postMessageQuestionCount: Optional[int] = Field(default=0)
    postMessageQuestionUnReadCount: Optional[int] = Field(default=0)
    postMessageAnswerCount: Optional[int] = Field(default=0)
    postMessageAnswerUnReadCount: Optional[int] = Field(default=0)
    postMessageCommentCount: Optional[int] = Field(default=0)
    postMessageCommentUnReadCount: Optional[int] = Field(default=0)
    savedFromPostMessage: Optional[bool] = Field(default=False)

    # Relations (Parse Server relations)
    postSocial: Optional[Dict[str, str]] = None
    chatMessages: Optional[Dict[str, str]] = None
    topAnswers: Optional[Dict[str, str]] = None
    postQuestions: Optional[Dict[str, str]] = None
    pinnedReplies: Optional[Dict[str, str]] = None
    memories: Optional[Dict[str, str]] = None
    selectedContext: Optional[Dict[str, str]] = None

    # Additional metadata fields
    hashtags: Optional[List[str]] = Field(default_factory=list)
    mentions: Optional[List[str]] = Field(default_factory=list)
    combined_createdAt: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='allow'
    )

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)

        # Transform pointer fields to use __type
        def transform_pointer(pointer_data):
            if pointer_data and isinstance(pointer_data, dict):
                if 'type' in pointer_data:
                    pointer_data['__type'] = pointer_data.pop('type')
            return pointer_data

        for field in ['user', 'workspace']:
            if field in data:
                data[field] = transform_pointer(data[field])

        # Transform file field (Parse File)
        if 'file' in data and isinstance(data['file'], dict):
            if 'type' in data['file']:
                data['file']['__type'] = data['file'].pop('type')

        # Normalize ACL if serialized oddly
        if 'ACL' in data and isinstance(data['ACL'], dict):
            if any(len(k) == 1 for k in data['ACL'].keys()):
                reconstructed_acl = {}
                current_key = []
                for k, v in sorted(data['ACL'].items()):
                    if k.startswith('role:'):
                        reconstructed_acl[k] = v
                    else:
                        current_key.append(k)
                        if k in ['[', ']']:
                            key = ''.join(current_key)
                            if key not in reconstructed_acl:
                                reconstructed_acl[key] = v
                            current_key = []
                data['ACL'] = reconstructed_acl

        return data

class PageVersionParseServer(BaseModel):
    """Model for PageVersion class in Parse Server."""
    objectId: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

    page: ParsePointer  # Pointer to Post
    author: ParsePointer  # Pointer to _User
    content: Dict[str, Any] = Field(default_factory=dict)
    versionType: Optional[str] = None
    processingMetadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    workspace: Optional[ParsePointer] = None

    # Optional convenience fields
    file: Optional[ParseFile] = None
    text: Optional[str] = None
    type: Optional[str] = Field(default="page")
    source: Optional[str] = Field(default="/document upload API")

    # Multi-tenant ids (compatibility)
    organizationId: Optional[str] = None
    namespaceId: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='allow'
    )

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)

        # Transform pointer fields
        def transform_pointer(pointer_data):
            if pointer_data and isinstance(pointer_data, dict):
                if 'type' in pointer_data:
                    pointer_data['__type'] = pointer_data.pop('type')
            return pointer_data

        for field in ['page', 'author', 'workspace']:
            if field in data:
                data[field] = transform_pointer(data[field])

        if 'file' in data and isinstance(data['file'], dict):
            if 'type' in data['file']:
                data['file']['__type'] = data['file'].pop('type')

        return data


class PostWithProviderResult(BaseModel):
    """
    Response model for fetching a Post with its provider result.
    
    This model wraps the PostParseServer with additional fields that are
    extracted from the Post for convenience in document processing workflows.
    """
    # The full Post object from Parse Server
    post: PostParseServer
    
    # Provider result (downloaded and decompressed if stored as file)
    provider_specific: Dict[str, Any] = Field(default_factory=dict)
    
    # Extracted fields for convenience
    provider_name: Optional[str] = None
    organization_id: Optional[str] = None
    namespace_id: Optional[str] = None
    workspace_id: Optional[str] = None
    upload_id: Optional[str] = None
    file_url: Optional[str] = None
    file_name: Optional[str] = None
    post_title: Optional[str] = None
    
    # Metadata from the Post
    file_metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)  # Extraction summary (if available)
    
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='allow'
    )


class BatchMemoryRequestParse(BaseModel):
    """
    Model for BatchMemoryRequest class in Parse Server.

    Stores batch memory request data to avoid Temporal gRPC payload limits.
    Follows the pattern from PostParseServer for large data storage.
    """
    objectId: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    ACL: Dict[str, Dict[str, bool]] = Field(default_factory=dict)
    className: Optional[str] = Field(default="BatchMemoryRequest")

    # Identifiers
    batchId: str
    requestId: Optional[str] = None

    # Multi-tenant context (Parse Pointers)
    organization: Optional[ParsePointer] = None
    namespace: Optional[ParsePointer] = None
    user: Optional[ParsePointer] = None
    workspace: Optional[ParsePointer] = None

    # Batch data storage
    batchDataFile: Optional[ParseFile] = None  # Compressed JSON with memories
    batchMetadata: Dict[str, Any] = Field(default_factory=dict)

    # Processing status
    status: str = "pending"  # pending|processing|completed|failed|partial_failure
    processedCount: int = 0
    successCount: int = 0
    failCount: int = 0
    totalMemories: int = 0

    # Temporal tracking
    workflowId: Optional[str] = None
    workflowRunId: Optional[str] = None

    # Webhook configuration
    webhookUrl: Optional[str] = None
    webhookSecret: Optional[str] = None
    webhookSent: bool = False

    # Timing metadata
    startedAt: Optional[datetime] = None
    completedAt: Optional[datetime] = None
    processingDurationMs: Optional[float] = None

    # Error tracking
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='allow'
    )

    def model_dump(self, *args, **kwargs):
        """Override to transform pointers to __type format"""
        data = super().model_dump(*args, **kwargs)

        # Transform pointer fields to use __type
        pointer_fields = ['organization', 'namespace', 'user', 'workspace']
        for field in pointer_fields:
            if field in data and isinstance(data[field], dict) and 'type' in data[field]:
                data[field]['__type'] = data[field].pop('type')
                data[field]['className'] = data[field].get('className', field.capitalize())

        # Transform file field
        if 'batchDataFile' in data and isinstance(data['batchDataFile'], dict):
            if 'type' in data['batchDataFile']:
                data['batchDataFile']['__type'] = data['batchDataFile'].pop('type')

        return data


class MemoryPredictionLog(BaseModel):
    """Model for MemoryPredictionLog class in Parse Server"""
    objectId: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    
    # Core Memory Context
    memoryItem: ParsePointer  # Pointer to Memory
    user: ParsePointer  # Pointer to _User
    workspace: ParsePointer  # Pointer to WorkSpace
    embeddingModel: Optional[str] = None
    
    # Prediction Generation (Indexing Phase)
    generatedSearchQueries: List[str] = Field(default_factory=list)  # 3 queries generated to find related memories
    predictedRelatedMemories: Optional[List[str]] = Field(default_factory=list)  # Array of Memory objectIds - will be handled as relation
    predictionConfidenceScores: List[float] = Field(default_factory=list)  # Parallel to predicted memories
    predictionMethod: Optional[str] = None  # "cosine_similarity", "topic_modeling", "user_behavior", "temporal"
    
    # Processing Metadata (Indexing Phase)
    predictionProcessingTimeMs: Optional[float] = None  # Total time for _index_memories_and_process
    relationshipCreationCount: Optional[float] = None  # Always 3 for top memories
    
    # Temporal Relationship Metrics (Indexing Phase)
    newToOldestMemoryAgeHours: Optional[float] = None  # Age difference to oldest retrieved memory
    newToNewestMemoryAgeHours: Optional[float] = None  # Age difference to newest retrieved memory
    newToMedianMemoryAgeHours: Optional[float] = None  # Age difference to median age of retrieved memories
    retrievedMemoriesAgeSpreadHours: Optional[float] = None  # Time span between oldest and newest retrieved memories
    temporalCoherenceScore: Optional[float] = None  # How clustered in time the retrieved memories are (0-1, higher = more clustered)
    
    # Validation Tracking (Updated Later)
    lastValidationUpdate: Optional[datetime] = None  # When predictions were last validated
    
    # Usage Patterns (Updated Later)
    previousQueryCount: Optional[float] = None
    previousQueryPatterns: List[str] = Field(default_factory=list)  # Types of queries that previously found this memory
    memoryAgeHours: Optional[float] = None  # How old the memory was when retrieved (calculated during validation)
    
    # Future Validation (Updated Later)
    actualFutureQueries: List[str] = Field(default_factory=list)  # Queries that later retrieved this memory
    predictionHitRate: Optional[float] = None  # % of predictions that were correct
    timeToFirstHit: Optional[float] = None  # Hours until first actual retrieval
    tierActuallyHit: Optional[str] = None  # "0", "1", "2" - which tier found it
    
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='forbid'
    )
    
    def model_dump(self, *args, **kwargs):
        """Override model_dump to ensure proper handling of pointer fields and exclude predictedRelatedMemories"""
        data = super().model_dump(*args, **kwargs)
        
        # Transform pointer fields to use __type
        pointer_fields = ['memoryItem', 'user', 'workspace']
        for field in pointer_fields:
            if field in data and isinstance(data[field], dict):
                if 'type' in data[field]:
                    data[field]['__type'] = data[field].pop('type')
        
        # Remove predictedRelatedMemories from the main data since it will be handled separately as a relation
        if 'predictedRelatedMemories' in data:
            data.pop('predictedRelatedMemories')
        
        return data

class QueryLog(BaseModel):
    """Model for QueryLog class in Parse Server"""
    objectId: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    
    # Core Query Context
    user: ParsePointer  # Pointer to _User
    workspace: ParsePointer  # Pointer to WorkSpace
    post: Optional[ParsePointer] = None  # Pointer to Post
    userMessage: Optional[ParsePointer] = None  # Pointer to PostMessage
    assistantMessage: Optional[ParsePointer] = None  # Pointer to PostMessage
    sessionId: Optional[str] = None
    queryText: str  # Required query text
    
    # Goal/UseCase/Step Classification (from request or generated)
    relatedGoals: Optional[List[ParsePointer]] = Field(default_factory=list)  # Pointers to Goal
    relatedUseCases: Optional[List[ParsePointer]] = Field(default_factory=list)  # Pointers to UseCase
    relatedSteps: Optional[List[ParsePointer]] = Field(default_factory=list)  # Pointers to Step
    goalClassificationScores: Optional[List[float]] = Field(default_factory=list)
    useCaseClassificationScores: Optional[List[float]] = Field(default_factory=list)
    stepClassificationScores: Optional[List[float]] = Field(default_factory=list)
    
    # Search Options
    rankingEnabled: Optional[bool] = None
    enabledAgenticGraph: Optional[bool] = None
    
    # Router Model Fields
    tierSequence: Optional[List[int]] = Field(default_factory=list)  # [0,1,2] - actual tiers accessed in order
    predictedTier: Optional[str] = None  # "0" - router's prediction
    actualTierHit: Optional[str] = None  # "1" - which tier actually had the answer
    tierPredictionAccuracy: Optional[bool] = None  # false - was prediction correct?
    tierPredictionConfidence: Optional[float] = None  # 0.85 - model confidence score
    
    # Performance Metrics
    totalProcessingTimeMs: Optional[float] = None
    retrievalLatencyMs: Optional[float] = None
    generationLatencyMs: Optional[float] = None
    
    # Cost Metrics
    queryEmbeddingTokens: Optional[int] = None  # Tokens used for query embedding
    retrievedMemoryTokens: Optional[int] = None  # Total tokens in retrieved memories
    llmPromptTokens: Optional[int] = None  # Tokens sent to LLM
    llmCompletionTokens: Optional[int] = None  # Tokens returned by LLM
    llmCostUsd: Optional[float] = None  # Total LLM cost for this query
    
    # Pricing and Billing
    pricingTier: Optional[str] = None  # "free", "pro", "enterprise"
    usageBasedBillingEnabled: Optional[bool] = None  # true if usage based billing is enabled
    llmCostPerThousandTokens: Optional[float] = None  # Current LLM pricing
    searchCostPerThousandTokens: Optional[float] = None  # Papr Memory search cost
    
    # System Context
    llmModelUsed: Optional[str] = None
    apiVersion: Optional[str] = None
    infrastructureRegion: Optional[str] = None
    
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra='allow'
    )
    
    @model_validator(mode="before")
    @classmethod
    def fix_parse_relations(cls, data):
        # Ensure relatedGoals, relatedUseCases, relatedSteps are lists, not relation dicts
        for field in ["relatedGoals", "relatedUseCases", "relatedSteps"]:
            v = data.get(field)
            if isinstance(v, dict) and v.get("__type") == "Relation":
                data[field] = []
        return data

    def model_dump(self, *args, **kwargs):
        """Override model_dump to ensure proper handling of pointer fields"""
        data = super().model_dump(*args, **kwargs)
        
        # Transform pointer fields to use __type
        pointer_fields = ['user', 'workspace', 'post', 'userMessage', 'assistantMessage']
        for field in pointer_fields:
            if field in data and isinstance(data[field], dict):
                if 'type' in data[field]:
                    data[field]['__type'] = data[field].pop('type')
        
        # Handle list of pointers for relations
        relation_fields = ['relatedGoals', 'relatedUseCases', 'relatedSteps']
        for field in relation_fields:
            if field in data and isinstance(data[field], list):
                for i, item in enumerate(data[field]):
                    if isinstance(item, dict) and 'type' in item:
                        data[field][i]['__type'] = item.pop('type')
        
        return data

class MemoryRetrievalLog(BaseModel):
    """Model for MemoryRetrievalLog class in Parse Server"""
    objectId: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

    # Core Context
    user: 'ParsePointer'  # Pointer to _User
    workspace: 'ParsePointer'  # Pointer to WorkSpace
    post: Optional['ParsePointer'] = None  # Pointer to Post
    sessionId: Optional[str] = None

    # QueryLog relation
    queryLog: 'ParsePointer'  # Pointer to QueryLog

    # Memory Relations
    retrievedMemories: Optional[List['ParsePointer']] = Field(default_factory=list)  # Relation to Memory
    citedMemories: Optional[List['ParsePointer']] = Field(default_factory=list)  # Relation to Memory

    # Arrays and metrics
    retrievedMemoryScores: Optional[List[float]] = Field(default_factory=list)
    memoryRetrievalTiers: Optional[List[int]] = Field(default_factory=list)
    totalMemoriesRetrieved: Optional[int] = None
    totalMemoriesCited: Optional[int] = None
    retrievalLatencyMs: Optional[float] = None
    embeddingLatencyMs: Optional[float] = None

    # Prediction/Grouping
    usedPredictedGrouping: Optional[bool] = None  # True if the answer came from a predicted group
    predictedGroupedMemories: Optional[List['ParsePointer']] = Field(default_factory=list)  # Relation to Memory
    groupedMemoriesDistribution: Optional[float] = None  # % of returned memories that were grouped
    predictionModelUsed: Optional[str] = None  # Model used if answer came from predicted group
    predictionAccuracyScore: Optional[float] = None

    # New fields for similarity and confidence scores
    retrievedMemorySimilarityScores: Optional[Dict[str, float]] = None
    retrievedMemoryConfidenceScores: Optional[Dict[str, float]] = None
    citedMemoryConfidenceScores: Optional[Dict[str, float]] = None  # NEW: feedback-driven confidence

    model_config = ConfigDict(
        from_attributes=True,
        extra='forbid',
        validate_by_name=True
    )

    @model_validator(mode="before")
    @classmethod
    def fix_parse_relations(cls, data):
        # Ensure relation fields are lists, not relation dicts
        for field in ["retrievedMemories", "citedMemories", "predictedGroupedMemories"]:
            v = data.get(field)
            if isinstance(v, dict) and v.get("__type") == "Relation":
                data[field] = []
        # Remove memoryPredictionLogs if it's a relation object
        if isinstance(data.get("memoryPredictionLogs"), dict) and data["memoryPredictionLogs"].get("__type") == "Relation":
            data.pop("memoryPredictionLogs")
        return data

    def model_dump(self, *args, **kwargs):
        """Override model_dump to ensure proper handling of pointer fields and relations"""
        data = super().model_dump(*args, **kwargs)
        # Transform pointer fields to use __type
        pointer_fields = ['user', 'workspace', 'post', 'queryLog']
        for field in pointer_fields:
            if field in data and isinstance(data[field], dict):
                if 'type' in data[field]:
                    data[field]['__type'] = data[field].pop('type')
        # Handle list of pointers for relations
        relation_fields = ['retrievedMemories', 'citedMemories', 'predictedGroupedMemories']
        for field in relation_fields:
            if field in data and isinstance(data[field], list):
                for i, item in enumerate(data[field]):
                    if isinstance(item, dict) and 'type' in item:
                        data[field][i]['__type'] = item.pop('type')
        return data

def remove_none(d):
    return {k: v for k, v in d.items() if v is not None}

def remove_none_recursive(obj):
    if isinstance(obj, dict):
        return {k: remove_none_recursive(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [remove_none_recursive(v) for v in obj if v is not None]
    else:
        return obj

class AgenticGraphLog(BaseModel):
    """Model for AgenticGraphLog class in Parse Server"""
    objectId: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    
    # Core Context
    user: 'ParsePointer'
    workspace: 'ParsePointer'
    post: Optional['ParsePointer'] = None
    sessionId: Optional[str] = None
    queryLog: 'ParsePointer'
    
    # Agentic Graph Fields
    naturalLanguageQueries: Optional[List[str]] = Field(default_factory=list)
    generatedCypherQueries: Optional[List[str]] = Field(default_factory=list)
    queryTypes: Optional[List[str]] = Field(default_factory=list)
    planningSteps: Optional[List[str]] = Field(default_factory=list)
    reasoningContext: Optional[List[str]] = Field(default_factory=list)
    planningStrategy: Optional[str] = None
    cypherExecutionResults: Optional[List[Any]] = Field(default_factory=list)
    traversalPaths: Optional[List[Any]] = Field(default_factory=list)
    nodesVisitedCounts: Optional[List[int]] = Field(default_factory=list)
    graphQueryComplexityScores: Optional[List[float]] = Field(default_factory=list)
    retrievedMemories: Optional[List['ParsePointer']] = Field(default_factory=list)  # Relation to Memory
    retrievedNodes: Optional[List[Any]] = Field(default_factory=list)
    retrievedNodeTypes: Optional[List[str]] = Field(default_factory=list)
    citedNodeIds: Optional[List[str]] = Field(default_factory=list)
    totalGraphLatencyMs: Optional[float] = None
    planningLatencyMs: Optional[float] = None
    cypherGenerationLatencyMs: Optional[float] = None
    graphExecutionLatencyMs: Optional[float] = None
    resultProcessingLatencyMs: Optional[float] = None
    totalQueriesExecuted: Optional[int] = None
    totalNodesRetrieved: Optional[int] = None
    totalNodesCited: Optional[int] = None
    agenticReasoningSuccess: Optional[bool] = None

    model_config = ConfigDict(
        from_attributes=True,
        extra='forbid',
        validate_by_name=True
    )

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        pointer_fields = ['user', 'workspace', 'post', 'queryLog']
        for field in pointer_fields:
            if field in data and isinstance(data[field], dict):
                if 'type' in data[field]:
                    data[field]['__type'] = data[field].pop('type')
        # Handle list of pointers for relations
        relation_fields = ['retrievedMemories']
        for field in relation_fields:
            if field in data and isinstance(data[field], list):
                for i, item in enumerate(data[field]):
                    if isinstance(item, dict) and 'type' in item:
                        data[field][i]['__type'] = item.pop('type')
        return data


class Chat(BaseModel):
    """Model for Chat class in Parse Server - manages conversation sessions"""
    objectId: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    ACL: Optional[Dict[str, Dict[str, bool]]] = None
    
    # Chat session management
    sessionId: str = Field(..., description="Unique session identifier for the conversation")
    title: Optional[str] = Field(None, description="Optional chat title")
    
    # Message tracking
    messageCount: int = Field(default=0, description="Total number of messages in this chat")
    lastProcessedMessageIndex: int = Field(default=0, description="Index of last processed message for batch analysis")
    
    # Processing status
    processingStatus: Optional[str] = Field(
        default="active", 
        description="Chat status: active, processing, completed, archived"
    )
    lastProcessedAt: Optional[datetime] = Field(None, description="When messages were last processed")
    
    # Relationships
    user: ParsePointer = Field(..., description="User who owns this chat")
    workspace: Optional[ParsePointer] = Field(None, description="Workspace context")
    
    # Multi-tenant fields
    organization: Optional[OrganizationPointer] = Field(None, description="Organization that owns this chat")
    namespace: Optional[NamespacePointer] = Field(None, description="Namespace this chat belongs to")
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional chat metadata")

    model_config = ConfigDict(
        from_attributes=True,
        extra='allow',
        validate_by_name=True
    )

    def model_dump(self, *args, **kwargs):
        def convert_dt(obj):
            if isinstance(obj, dict):
                return {k: convert_dt(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dt(v) for v in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj
        data = super().model_dump(*args, **kwargs)
        return convert_dt(data)


class PostMessage(BaseModel):
    """Model for PostMessage class in Parse Server - matches existing GraphQL schema"""
    objectId: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    ACL: Optional[Dict[str, Dict[str, bool]]] = None
    
    # Core message fields (from GraphQL schema)
    message: Optional[str] = Field(None, description="The chat message content")
    content: Optional[str] = Field(None, description="Alternative content field")
    assistantResponse: Optional[str] = Field(None, description="Assistant response content")
    messageRole: Optional[str] = Field(None, description="Role of the message sender (user or assistant)")
    title: Optional[str] = Field(None, description="Message title")
    
    # AI/LLM fields
    model: Optional[str] = Field(None, description="AI model used")
    promptTokens: Optional[int] = Field(None, description="Number of prompt tokens")
    userPromptTokens: Optional[int] = Field(None, description="Number of user prompt tokens")
    completionTokens: Optional[int] = Field(None, description="Number of completion tokens")
    inputCosts: Optional[float] = Field(None, description="Input costs")
    outputCosts: Optional[float] = Field(None, description="Output costs")
    totalCosts: Optional[float] = Field(None, description="Total costs")
    
    # Content fields
    highlightedText: Optional[str] = Field(None, description="Highlighted text")
    highlightedTextId: Optional[str] = Field(None, description="Highlighted text ID")
    imageUrls: Optional[List[str]] = Field(default_factory=list, description="Image URLs")
    
    # Processing status for batch analysis
    processingStatus: Optional[str] = Field(
        default="pending", 
        description="Status: pending, processing, completed, failed, stored_only"
    )
    
    # Relationships (from GraphQL schema)
    user: ParsePointer = Field(..., description="User who sent the message")
    userTo: Optional[ParsePointer] = Field(None, description="User message is directed to")
    workspace: Optional[ParsePointer] = Field(None, description="Workspace context")
    post: Optional[ParsePointer] = Field(None, description="Related post")
    parentPostMessage: Optional[ParsePointer] = Field(None, description="Parent message for threading")
    replyToUserPostMessage: Optional[ParsePointer] = Field(None, description="Message this is replying to")
    chat: Optional[ParsePointer] = Field(None, description="Chat session this message belongs to")
    
    # Memory relationship
    memoriesUsed: Optional[List[ParsePointer]] = Field(default_factory=list, description="Memories used/created")
    
    # Message threading
    childPostMessageCount: Optional[int] = Field(default=0, description="Number of child messages")
    
    # Multi-tenant fields
    organization: Optional[OrganizationPointer] = Field(None, description="Organization that owns this message")
    namespace: Optional[NamespacePointer] = Field(None, description="Namespace this message belongs to")
    
    # Error tracking
    processingError: Optional[str] = Field(None, description="Error message if processing failed")
    processingAttempts: Optional[int] = Field(default=0, description="Number of processing attempts")

    model_config = ConfigDict(
        from_attributes=True,
        extra='allow',
        validate_by_name=True
    )

    def model_dump(self, *args, **kwargs):
        def convert_dt(obj):
            if isinstance(obj, dict):
                return {k: convert_dt(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dt(v) for v in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj
        data = super().model_dump(*args, **kwargs)
        return convert_dt(data)


class UserFeedbackLog(BaseModel):
    """Model for UserFeedbackLog class in Parse Server"""
    objectId: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    
    # Core Context
    queryLog: 'ParsePointer'
    user: 'ParsePointer'
    workspace: 'ParsePointer'
    post: Optional['ParsePointer'] = None
    sessionId: Optional[str] = None
    userMessage: Optional[ParsePointer] = None  # Pointer to PostMessage
    assistantMessage: Optional[ParsePointer] = None  # Pointer to PostMessage
    
    # Feedback Fields
    feedbackType: FeedbackType
    feedbackValue: Optional[str] = None
    feedbackScore: Optional[float] = None
    feedbackText: Optional[str] = None
    feedbackSource: FeedbackSource
    citedMemoryIds: Optional[List[str]] = Field(default_factory=list)
    citedNodeIds: Optional[List[str]] = Field(default_factory=list)
    feedbackProcessed: Optional[bool] = None
    feedbackImpact: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True,
        extra='forbid',
        validate_by_name=True
    )

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        pointer_fields = ['queryLog', 'user', 'workspace', 'post']
        for field in pointer_fields:
            if field in data and isinstance(data[field], dict):
                if 'type' in data[field]:
                    data[field]['__type'] = data[field].pop('type')
        return data

