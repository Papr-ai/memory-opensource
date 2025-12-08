"""
Interaction and API Operation Tracking Models

This module defines models for tracking:
1. Interaction - Monthly aggregated counts for rate limiting (Parse Server)
2. APIOperationLog - Detailed operation logs for analytics (MongoDB Time Series)
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class InteractionType(str, Enum):
    """
    Type of interaction being tracked
    
    LLM Interactions:
    - MINI: Regular LLM calls (search, chat with mini model)
    - PREMIUM: Premium LLM calls (chat with premium model)
    
    API Operations:
    - API_OPERATION: API calls (memory CRUD, user operations, etc.)
    """
    MINI = "mini"  # LLM: Mini model interactions (includes search)
    PREMIUM = "premium"  # LLM: Premium model interactions
    API_OPERATION = "api_operation"  # API: All API operations


class HTTPMethod(str, Enum):
    """HTTP methods for API operations"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class APIRoute(str, Enum):
    """API routes being tracked"""
    # Memory routes
    MEMORY = "v1/memory"
    MEMORY_SEARCH = "v1/memory/search"
    MEMORY_BATCH = "v1/memory/batch"
    
    # User routes
    USER = "v1/user"
    
    # Other routes can be added as needed
    UNKNOWN = "unknown"


# ============================================================================
# PARSE SERVER MODELS (Monthly Aggregates)
# ============================================================================

class InteractionPointer(BaseModel):
    """Pointer to Interaction object in Parse Server"""
    objectId: str
    type: str = Field(default="Pointer", alias="__type")
    className: Literal["Interaction"] = "Interaction"
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        extra='forbid'
    )


class Interaction(BaseModel):
    """
    Monthly aggregated interaction counts for rate limiting
    
    Stored in Parse Server for ACL support and Parse Dashboard visibility.
    Used by check_interaction_limits() for monthly quota enforcement.
    """
    objectId: Optional[str] = Field(default=None, description="Parse objectId")
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    
    # Who (Pointers)
    user: dict = Field(..., description="Pointer to _User")
    workspace: dict = Field(..., description="Pointer to WorkSpace")
    organization: Optional[dict] = Field(default=None, description="Pointer to Organization (for multi-tenant)")
    company: Optional[dict] = Field(default=None, description="Pointer to Company (legacy)")
    subscription: Optional[dict] = Field(default=None, description="Pointer to Subscription")
    
    # What
    type: InteractionType = Field(..., description="Type of interaction: mini, premium, or api_operation")
    
    # When (for monthly aggregation)
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    year: int = Field(..., description="Year (e.g., 2025)")
    
    # Count
    count: int = Field(default=0, description="Number of interactions this month")
    
    # API Operation specific (only for type=api_operation)
    route: Optional[str] = Field(default=None, description="API route (e.g., 'v1/memory', 'v1/user')")
    method: Optional[str] = Field(default=None, description="HTTP method (GET, POST, PUT, DELETE)")
    isMemoryOperation: Optional[bool] = Field(
        default=None, 
        description="True if this counts against memory operation limits"
    )
    
    # Parse Server ACL
    ACL: Optional[dict] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None
        }
    )
    
    @classmethod
    def create_llm_interaction(
        cls,
        user_id: str,
        workspace_id: str,
        interaction_type: Literal["mini", "premium"],
        month: int,
        year: int,
        organization_id: Optional[str] = None,
        company_id: Optional[str] = None,
        subscription_id: Optional[str] = None
    ) -> "Interaction":
        """Create an LLM interaction record"""
        interaction = {
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            },
            "workspace": {
                "__type": "Pointer",
                "className": "WorkSpace",
                "objectId": workspace_id
            },
            "type": interaction_type,
            "month": month,
            "year": year,
            "count": 0
        }
        
        if organization_id:
            interaction["organization"] = {
                "__type": "Pointer",
                "className": "Organization",
                "objectId": organization_id
            }
        
        if company_id:
            interaction["company"] = {
                "__type": "Pointer",
                "className": "Company",
                "objectId": company_id
            }
        
        if subscription_id:
            interaction["subscription"] = {
                "__type": "Pointer",
                "className": "Subscription",
                "objectId": subscription_id
            }
        
        return cls(**interaction)
    
    @classmethod
    def create_api_operation_interaction(
        cls,
        user_id: str,
        workspace_id: str,
        route: str,
        method: str,
        is_memory_operation: bool,
        month: int,
        year: int,
        organization_id: Optional[str] = None,
        subscription_id: Optional[str] = None
    ) -> "Interaction":
        """Create an API operation interaction record"""
        interaction = {
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            },
            "workspace": {
                "__type": "Pointer",
                "className": "WorkSpace",
                "objectId": workspace_id
            },
            "type": InteractionType.API_OPERATION.value,
            "route": route,
            "method": method,
            "isMemoryOperation": is_memory_operation,
            "month": month,
            "year": year,
            "count": 0
        }
        
        if organization_id:
            interaction["organization"] = {
                "__type": "Pointer",
                "className": "Organization",
                "objectId": organization_id
            }
        
        if subscription_id:
            interaction["subscription"] = {
                "__type": "Pointer",
                "className": "Subscription",
                "objectId": subscription_id
            }
        
        return cls(**interaction)


# ============================================================================
# TIME SERIES MODELS (Detailed Logs)
# ============================================================================

class APIOperationLog(BaseModel):
    """
    Detailed API operation log for analytics (MongoDB Time Series)
    
    This provides granular tracking of every API call with full context.
    Optimized for time-based queries and analytics.
    
    Storage: MongoDB Time Series Collection
    Retention: 90 days (configurable)
    """
    # Time (required for time series)
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the operation occurred (UTC)"
    )
    
    # Who
    user_id: str = Field(..., description="User who made the call")
    workspace_id: str = Field(..., description="Workspace context")
    organization_id: Optional[str] = Field(default=None, description="Organization (for multi-tenant)")
    developer_id: Optional[str] = Field(default=None, description="API key owner (if different from user)")
    
    # What
    route: str = Field(..., description="API route (e.g., 'v1/memory', 'v1/memory/search')")
    method: HTTPMethod = Field(..., description="HTTP method")
    operation_type: str = Field(..., description="Operation type (e.g., 'add_memory', 'search', 'update_memory')")
    
    # Memory operation specific
    is_memory_operation: bool = Field(
        default=False,
        description="True if counts against memory operation limits"
    )
    memory_id: Optional[str] = Field(default=None, description="Memory ID if applicable")
    batch_size: Optional[int] = Field(default=None, description="Batch size if batch operation")
    
    # Performance
    latency_ms: Optional[float] = Field(default=None, description="Response time in milliseconds")
    status_code: Optional[int] = Field(default=None, description="HTTP status code")
    
    # Context
    api_key: Optional[str] = Field(default=None, description="API key used (hashed)")
    client_type: Optional[str] = Field(default=None, description="Client type (e.g., 'papr_plugin', 'sdk')")
    ip_address: Optional[str] = Field(default=None, description="Client IP address")
    user_agent: Optional[str] = Field(default=None, description="User agent")
    
    # Metadata
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None
        }
    )
    
    def to_mongo_doc(self) -> dict:
        """Convert to MongoDB document format"""
        return self.model_dump(exclude_none=True)


# ============================================================================
# HELPER TYPES
# ============================================================================

class InteractionSummary(BaseModel):
    """Summary of interactions for a time period"""
    mini_count: int = 0
    premium_count: int = 0
    api_operation_count: int = 0
    memory_operation_count: int = 0
    total_count: int = 0
    
    model_config = ConfigDict(from_attributes=True)

