# Multi-Tenant Pydantic Models

This document outlines the new Pydantic models added to `models/parse_server.py` for multi-tenant architecture support.

## Overview

We've added multi-tenant support while maintaining **backward compatibility** with existing code. All existing classes (`Workspace`, `_User`, `DeveloperUser`, `Memory`) remain functional.

## New Enums

### UserType
```python
class UserType(str, Enum):
    """User type classification for multi-tenant architecture"""
    DEVELOPER = "DEVELOPER"        # API key owner, organization owner
    END_USER = "END_USER"          # Developer's customer
    TEAM_MEMBER = "TEAM_MEMBER"    # Member of developer's team
```

### EnvironmentType
```python
class EnvironmentType(str, Enum):
    """Environment types for namespaces"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
```

## New Models

### 1. Organization
**Represents a tenant/company using Papr**

```python
class Organization(BaseModel):
    objectId: Optional[str]           # Parse objectId
    createdAt: Optional[datetime]
    updatedAt: Optional[datetime]
    
    # Core fields
    name: str                         # Organization name
    owner_user_id: str                # User ID of owner (points to _User)
    team_members: List[str]           # List of team member user IDs
    
    # Subscription & Billing
    subscription_id: Optional[str]    # Stripe subscription ID
    plan_tier: str                    # trial, starter, growth, pro, business_plus, enterprise
    
    # Settings
    settings: Dict[str, Any]          # {default_namespace, allowed_namespaces}
    
    # ACL
    ACL: Optional[Dict[str, Dict[str, bool]]]
```

**Example:**
```python
org = Organization(
    name="Acme Corp",
    owner_user_id="user_abc123",
    team_members=["user_abc123", "user_def456"],
    plan_tier="pro",
    settings={
        "default_namespace": "ns_acme_production",
        "allowed_namespaces": ["ns_acme_production", "ns_acme_dev"]
    }
)
```

### 2. Namespace
**Isolated environment within an organization**

```python
class Namespace(BaseModel):
    objectId: Optional[str]           # Parse objectId
    createdAt: Optional[datetime]
    updatedAt: Optional[datetime]
    
    # Core fields
    name: str                         # e.g., "acme-production"
    organization_id: str              # Organization ID this belongs to
    environment_type: EnvironmentType # development, staging, production
    
    # Status
    is_active: bool                   # Whether namespace is active
    
    # Rate limits (can override organization defaults)
    rate_limits: Dict[str, Optional[int]]  # {memories_per_month, api_calls_per_day}
    
    # ACL
    ACL: Optional[Dict[str, Dict[str, bool]]]
```

**Example:**
```python
namespace = Namespace(
    name="acme-production",
    organization_id="org_abc123",
    environment_type=EnvironmentType.PRODUCTION,
    is_active=True,
    rate_limits={
        "memories_per_month": None,  # Unlimited
        "api_calls_per_day": 10000
    }
)
```

### 3. APIKey
**API Key for authenticating to a specific namespace**

```python
class APIKey(BaseModel):
    objectId: Optional[str]           # Parse objectId
    createdAt: Optional[datetime]
    updatedAt: Optional[datetime]
    
    # Core fields
    key: str                          # The actual API key
    name: str                         # "Production Key", "Test Key"
    
    # Tenant hierarchy
    namespace_id: str                 # Namespace ID this key belongs to
    organization_id: str              # Organization ID (for quick lookup)
    
    # Metadata
    environment: str                  # production, development, staging
    permissions: List[str]            # ["read", "write", "delete"]
    
    # Status
    is_active: bool                   # Whether key is active
    last_used_at: Optional[datetime]  # Last usage timestamp
    
    # ACL
    ACL: Optional[Dict[str, Dict[str, bool]]]
```

**Example:**
```python
api_key = APIKey(
    key="pk_live_abc123xyz",
    name="Production API Key",
    namespace_id="ns_acme_production",
    organization_id="org_abc123",
    environment="production",
    permissions=["read", "write"],
    is_active=True
)
```

## Updated Pointer Classes

### OrganizationPointer
```python
class OrganizationPointer(BaseModel):
    objectId: str
    type: str = Field(default="Pointer", alias="__type")
    className: Literal["Organization"] = "Organization"
```

### NamespacePointer
```python
class NamespacePointer(BaseModel):
    objectId: str
    type: str = Field(default="Pointer", alias="__type")
    className: Literal["Namespace"] = "Namespace"
```

## Updated Existing Models

### ParseUserPointer (Updated)
**Added multi-tenant fields:**

```python
class ParseUserPointer(BaseModel):
    # ... existing fields ...
    
    # Multi-tenant fields (NEW)
    user_type: Optional[UserType]                # DEVELOPER, END_USER, TEAM_MEMBER
    organization_id: Optional[str]               # If DEVELOPER or TEAM_MEMBER
    developer_organization_id: Optional[str]     # If END_USER
    external_id: Optional[str]                   # Developer's ID for end user
    
    # Deprecated (kept for backward compatibility)
    isDeveloper: Optional[bool]
```

### DeveloperUserPointer (Updated)
**Added multi-tenant fields:**

```python
class DeveloperUserPointer(BaseModel):
    # ... existing fields ...
    
    # Multi-tenant fields (NEW)
    organization_id: Optional[str]    # Organization ID end user belongs to
    namespace_id: Optional[str]       # Namespace ID end user belongs to
```

### ParseStoredMemory (Updated)
**Added multi-tenant fields:**

```python
class ParseStoredMemory(BaseModel):
    # ... existing fields ...
    
    # Multi-tenant fields (NEW)
    organization_id: Optional[str]    # Organization that owns this memory
    namespace_id: Optional[str]       # Namespace this memory belongs to
```

### Memory (Updated)
**Public-facing memory model with multi-tenant fields:**

```python
class Memory(BaseModel):
    # ... existing fields ...
    
    # Multi-tenant fields (NEW)
    organization_id: Optional[str]    # Organization that owns this memory
    namespace_id: Optional[str]       # Namespace this memory belongs to
```

### MemoryParseServer (Updated)
**Added multi-tenant fields:**

```python
class MemoryParseServer(BaseModel):
    # ... existing fields ...
    
    # Multi-tenant fields (NEW)
    organization_id: Optional[str]    # Organization that owns this memory
    namespace_id: Optional[str]       # Namespace this memory belongs to
```

### MemoryParseServerUpdate (Updated)
**Added multi-tenant fields:**

```python
class MemoryParseServerUpdate(BaseModel):
    # ... existing fields ...
    
    # Multi-tenant fields (NEW)
    organization_id: Optional[str]    # Organization that owns this memory
    namespace_id: Optional[str]       # Namespace this memory belongs to
```

## Database Schema Summary

### New Collections in Parse Server / MongoDB

```
Organization
  - objectId: string (primary key)
  - name: string
  - owner_user_id: string (→ _User)
  - team_members: [string] (→ _User[])
  - subscription_id: string (→ Stripe)
  - plan_tier: string
  - settings: object
  - ACL: object

Namespace
  - objectId: string (primary key)
  - name: string
  - organization_id: string (→ Organization)
  - environment_type: enum
  - is_active: boolean
  - rate_limits: object
  - ACL: object

APIKey
  - objectId: string (primary key)
  - key: string (indexed, unique)
  - name: string
  - namespace_id: string (→ Namespace)
  - organization_id: string (→ Organization)
  - environment: string
  - permissions: [string]
  - is_active: boolean
  - last_used_at: datetime
  - ACL: object
```

### Updated Collections

```
_User (Updated)
  - ... existing fields ...
  - user_type: enum (DEVELOPER, END_USER, TEAM_MEMBER)
  - organization_id: string (→ Organization, if DEVELOPER/TEAM_MEMBER)
  - developer_organization_id: string (→ Organization, if END_USER)
  - external_id: string (if END_USER)
  - isDeveloper: boolean (DEPRECATED)

DeveloperUser (Updated)
  - ... existing fields ...
  - organization_id: string (→ Organization)
  - namespace_id: string (→ Namespace)

Memory (Updated)
  - ... existing fields ...
  - organization_id: string (→ Organization)
  - namespace_id: string (→ Namespace)
```

## Hierarchical Relationships

```
Organization (Tenant/Company)
  └── Namespace (Environment: prod, dev, staging)
       ├── APIKey (Authentication)
       ├── DeveloperUser (End Users)
       │    └── Memory (User's memories)
       └── Memory (All memories in namespace)
```

## Usage Examples

### Creating a Developer Organization

```python
from models.parse_server import Organization, Namespace, APIKey, UserType

# 1. Create organization
org = Organization(
    name="Acme Corp",
    owner_user_id="user_abc123",
    plan_tier="pro"
)

# 2. Create production namespace
namespace = Namespace(
    name="acme-production",
    organization_id=org.objectId,
    environment_type=EnvironmentType.PRODUCTION,
    is_active=True
)

# 3. Generate API key
api_key = APIKey(
    key="pk_live_abc123",
    name="Production Key",
    namespace_id=namespace.objectId,
    organization_id=org.objectId,
    environment="production"
)
```

### Creating an End User

```python
from models.parse_server import ParseUserPointer, DeveloperUserPointer

# 1. Create _User record
user = ParseUserPointer(
    objectId="user_xyz789",
    user_type=UserType.END_USER,
    developer_organization_id="org_abc123",  # Links to developer
    external_id="customer_123"  # Developer's ID for this user
)

# 2. Create DeveloperUser record
dev_user = DeveloperUserPointer(
    objectId="devuser_xyz",
    external_id="customer_123",
    organization_id="org_abc123",
    namespace_id="ns_acme_production"
)
```

### Creating a Memory with Multi-Tenant Fields

```python
from models.parse_server import MemoryParseServer

memory = MemoryParseServer(
    content="Meeting notes from Q4 planning",
    type="TextMemoryItem",
    user=ParseUserPointer(objectId="user_xyz789"),
    organization_id="org_abc123",  # NEW
    namespace_id="ns_acme_production",  # NEW
    ACL={
        "org_abc123": {"read": True, "write": True}
    }
)
```

## Query Patterns

### Developer Dashboard - All Memories

```python
# Get all memories for organization (across all end users)
memories = db.Memory.find({
    "organization_id": "org_abc123"
})
```

### Developer Dashboard - Namespace Filtered

```python
# Get memories for specific namespace (e.g., production only)
memories = db.Memory.find({
    "organization_id": "org_abc123",
    "namespace_id": "ns_acme_production"
})
```

### End User Query

```python
# Get memories for specific end user
memories = db.Memory.find({
    "user_id": "user_xyz789"
})
```

## Migration Path

### Phase 1: Add Fields (Non-Breaking) ✅ COMPLETE
- ✅ Created new Pydantic models
- ✅ Added optional fields to existing models
- ✅ All fields are `Optional` for backward compatibility

### Phase 2: Data Migration (Next Step)
```bash
# Run migration script
poetry run python scripts/migrate_to_multi_tenant.py
```

This will:
1. Create `Organization` for each existing developer
2. Create default `Namespace` for each organization
3. Migrate existing `APIKey` records
4. Backfill `organization_id` and `namespace_id` on existing memories

### Phase 3: Update API Endpoints
- Update auth middleware to resolve API key → namespace → org
- Update memory endpoints to use hierarchy
- Add namespace filtering to queries

### Phase 4: Enforce in New Code
- Start requiring `organization_id` and `namespace_id` for new memories
- Update UI to show namespace selector
- Add organization management endpoints

## Backward Compatibility

✅ **All existing code continues to work**
- Old memories without `organization_id`/`namespace_id` still accessible
- Existing ACLs still enforced
- `Workspace` concept still supported
- `isDeveloper` flag still present (deprecated)

The migration is **additive** - we're adding new capabilities while keeping the old system functional.

## Next Steps

1. **Run Migration Script**: `poetry run python scripts/migrate_to_multi_tenant.py`
2. **Update Auth Middleware**: Resolve API keys to org/namespace
3. **Update Memory Routes**: Add tenant filtering
4. **Create Dashboard APIs**: Organization management endpoints
5. **Add Namespace UI**: Namespace selector in developer dashboard

## Benefits

✅ **Logical Separation**: Each organization's data is isolated via filtering
✅ **Single Database**: No need to split databases
✅ **Fast Queries**: Proper indexes make tenant-filtered queries fast
✅ **Easy Analytics**: Cross-tenant analytics in single database
✅ **Scalable**: Handles millions of tenants efficiently
✅ **Backward Compatible**: Existing code keeps working

