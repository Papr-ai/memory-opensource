# Multi-Tenant Schema Design for Papr Memory

## Overview

This document outlines the complete multi-tenant schema design for Papr Memory using Parse Server proper pointer and relation types.

## Design Principles

1. **Parse Server Native**: Use Parse Pointers and Relations, not string IDs
2. **Backward Compatible**: Keep existing `Workspace` and `_User` classes working
3. **Hierarchical**: Organization → Namespace → Resources
4. **Flexible Rate Limits**: Org-level defaults, namespace-level overrides
5. **Role-Based Access**: Map workspace roles to organization roles

## Rate Limits (Based on Papr Pricing Tiers)

Based on Papr's actual pricing page (https://papr.ai/pricing):

### Papr Pricing Tiers

| Tier | Memory Operations/Month | Storage | Active Memories | Rate Limit/Min | Price |
|------|------------------------|---------|-----------------|----------------|-------|
| **Developer** | 1,000 | 1GB | 2,500 | 10 | Free |
| **Starter** | 50,000 | 10GB | 100,000 | 30 | $100/mo |
| **Growth** | 750,000 | 100GB | 1,000,000 | 100 | $500/mo |
| **Enterprise** | Unlimited | Unlimited | Unlimited | 500 | Custom |

### Definitions

- **Memory Operations**: API calls for creating, reading, updating, or deleting memories
- **Storage**: Total storage capacity for memory content and metadata
- **Active Memories**: Maximum number of memories that can exist (not deleted)
- **Rate Limit/Min**: Burst protection - maximum requests per minute

### Our Rate Limit Strategy

**Organization Level (Default)**:
- `max_memory_operations_per_month`: Monthly memory operation limit (create/read/update/delete)
- `max_storage_gb`: Total storage capacity in gigabytes
- `max_active_memories`: Maximum number of non-deleted memories
- `rate_limit_per_minute`: Burst protection

**Namespace Level (Override)**:
- Can override org defaults (useful for production vs dev namespaces)
- `null` = inherit from organization
- Specific number = override

**Example** (Growth Tier):
```javascript
Organization {
  plan_tier: "growth",
  rate_limits: {
    max_memory_operations_per_month: 750000,  // 750K operations
    max_storage_gb: 100,  // 100GB storage
    max_active_memories: 1000000,  // 1M active memories
    rate_limit_per_minute: 100
  }
}

Namespace (production) {
  rate_limits: {
    max_memory_operations_per_month: null,  // Inherit 750K from org
    max_storage_gb: null,  // Inherit 100GB from org
    max_active_memories: null,  // Inherit 1M from org
    rate_limit_per_minute: 200  // Override: higher for production
  }
}

Namespace (development) {
  rate_limits: {
    max_memory_operations_per_month: 10000,  // Override: lower for testing
    max_storage_gb: 1,  // Override: 1GB for dev
    max_active_memories: 10000,  // Override: lower for testing
    rate_limit_per_minute: null  // Inherit 100 from org
  }
}
```

## Schema Design (Parse Server Format)

### 1. Organization Class

**Purpose**: Represents a tenant/company using Papr

```javascript
{
  // Parse Standard Fields
  "objectId": "org_abc123",  // Auto-generated
  "_created_at": ISODate("2025-10-01T00:00:00Z"),
  "_updated_at": ISODate("2025-10-01T00:00:00Z"),
  "_acl": {
    "86cRDG7c4z": {"read": true, "write": true},
    "user123": {"read": true, "write": false}
  },
  "_rperm": ["86cRDG7c4z", "user123"],
  "_wperm": ["86cRDG7c4z"],
  
  // Core Fields with Parse Pointers
  "name": "Acme Corp",
  "owner": {
    "__type": "Pointer",
    "className": "_User",
    "objectId": "86cRDG7c4z"
  },
  "workspace": {  // Backward compatibility link
    "__type": "Pointer",
    "className": "WorkSpace",
    "objectId": "EaAJm7b1zN"
  },
  
  // Subscription & Billing
  "subscription": {
    "__type": "Pointer",
    "className": "Subscription",
    "objectId": "sub_xxx"
  },
  "plan_tier": "growth",  // developer, starter, growth, enterprise
  
  // Rate Limits (Org-level defaults - Growth tier)
  "rate_limits": {
    "max_memory_operations_per_month": 750000,  // 750K memory operations
    "max_storage_gb": 100,  // 100GB storage
    "max_active_memories": 1000000,  // 1M active memories
    "rate_limit_per_minute": 100
  },
  
  // Default Namespace Pointer
  "default_namespace": {
    "__type": "Pointer",
    "className": "Namespace",
    "objectId": "ns_org_abc123_production"
  }
}

// Parse Relations (many-to-many, managed via REST API)
// team_members: Relation<_User>
// allowed_namespaces: Relation<Namespace>
```

**Indexes**:
```javascript
db.Organization.createIndex({"_p_owner": 1})
db.Organization.createIndex({"_p_workspace": 1})
db.Organization.createIndex({"plan_tier": 1})
```

### 2. Namespace Class

**Purpose**: Isolated environment within an organization (e.g., production, staging, development)

```javascript
{
  // Parse Standard Fields
  "objectId": "ns_org_abc123_production",
  "_created_at": ISODate("2025-10-01T00:00:00Z"),
  "_updated_at": ISODate("2025-10-01T00:00:00Z"),
  "_acl": {
    "86cRDG7c4z": {"read": true, "write": true}
  },
  "_rperm": ["86cRDG7c4z"],
  "_wperm": ["86cRDG7c4z"],
  
  // Core Fields with Parse Pointers
  "name": "acme-production",
  "organization": {
    "__type": "Pointer",
    "className": "Organization",
    "objectId": "org_abc123"
  },
  "environment_type": "production",  // development, staging, production
  
  // Status
  "is_active": true,
  
  // Rate Limits (can override org defaults)
  "rate_limits": {
    "max_memory_operations_per_month": null,  // null = inherit from org (750K)
    "max_storage_gb": null,  // null = inherit from org (100GB)
    "max_active_memories": null,  // null = inherit from org (1M)
    "rate_limit_per_minute": 200  // override: higher for production
  }
}
```

**Indexes**:
```javascript
db.Namespace.createIndex({"_p_organization": 1})
db.Namespace.createIndex({"environment_type": 1})
db.Namespace.createIndex({"is_active": 1})
```

### 3. APIKey Class

**Purpose**: Authentication key scoped to a namespace

```javascript
{
  // Parse Standard Fields
  "objectId": "ak_xyz789",
  "_created_at": ISODate("2025-10-01T00:00:00Z"),
  "_updated_at": ISODate("2025-10-01T00:00:00Z"),
  "_acl": {
    "86cRDG7c4z": {"read": true, "write": true}
  },
  "_rperm": ["86cRDG7c4z"],
  "_wperm": ["86cRDG7c4z"],
  
  // Core Fields
  "key": "tED0YUF0XZqH6xFHJs7ezGQvA1MNcXsq",  // The actual API key
  "name": "Production API Key",
  
  // Parse Pointers
  "namespace": {
    "__type": "Pointer",
    "className": "Namespace",
    "objectId": "ns_org_abc123_production"
  },
  "organization": {
    "__type": "Pointer",
    "className": "Organization",
    "objectId": "org_abc123"
  },
  
  // Metadata
  "environment": "production",
  "permissions": ["read", "write", "delete"],
  
  // Status
  "is_active": true,
  "last_used_at": ISODate("2025-10-01T12:00:00Z")
}
```

**Indexes**:
```javascript
db.APIKey.createIndex({"key": 1}, {unique: true})
db.APIKey.createIndex({"_p_namespace": 1})
db.APIKey.createIndex({"_p_organization": 1})
db.APIKey.createIndex({"is_active": 1})
```

### 4. _User Class (Updated)

**Purpose**: Represents all users - developers, team members, and end users

```javascript
{
  // Existing Parse User Fields
  "objectId": "86cRDG7c4z",
  "username": "shawkat@papr.ai",
  "email": "shawkat@papr.ai",
  "_created_at": ISODate("2022-08-18T01:09:51.613Z"),
  "_updated_at": ISODate("2025-10-01T07:49:13.643Z"),
  
  // Existing Fields
  "displayName": "Shawkat Kabbara",
  "isSelectedWorkspaceFollower": {
    "__type": "Pointer",
    "className": "workspace_follower",
    "objectId": "6mu7xHjNXH"
  },
  
  // NEW: Multi-tenant fields with Parse Pointers
  "user_type": "DEVELOPER",  // DEVELOPER, END_USER, TEAM_MEMBER
  
  // If DEVELOPER or TEAM_MEMBER
  "organization": {
    "__type": "Pointer",
    "className": "Organization",
    "objectId": "org_86cRDG7c4z"
  },
  
  // If END_USER (developer's customer)
  "developer_organization": {
    "__type": "Pointer",
    "className": "Organization",
    "objectId": "org_creator_id"
  },
  "external_id": "user_12345",  // Developer's ID for this user
  
  // Deprecated (kept for backward compatibility)
  "isDeveloper": true,
  "userAPIkey": "tED0YUF0XZqH6xFHJs7ezGQvA1MNcXsq",
  "organization_id": "org_86cRDG7c4z",  // DEPRECATED: use organization pointer
  
  // ACL
  "_acl": {
    "86cRDG7c4z": {"read": true, "write": true}
  }
}
```

**Indexes**:
```javascript
db._User.createIndex({"user_type": 1})
db._User.createIndex({"_p_organization": 1})
db._User.createIndex({"_p_developer_organization": 1})
db._User.createIndex({"external_id": 1})
```

### 5. DeveloperUser Class (Updated)

**Purpose**: Links end users to developers (backward compatibility)

```javascript
{
  "objectId": "devuser_123",
  "_created_at": ISODate("2025-01-01T00:00:00Z"),
  "_updated_at": ISODate("2025-01-01T00:00:00Z"),
  
  // Parse Pointer to the end user's _User record
  "user": {
    "__type": "Pointer",
    "className": "_User",
    "objectId": "user_xyz789"
  },
  
  // NEW: Multi-tenant with Parse Pointers
  "organization": {
    "__type": "Pointer",
    "className": "Organization",
    "objectId": "org_abc123"
  },
  "namespace": {
    "__type": "Pointer",
    "className": "Namespace",
    "objectId": "ns_org_abc123_production"
  },
  
  // External ID from developer
  "external_id": "customer_12345",
  "metadata": {
    "email": "customer@example.com",
    "custom_field": "value"
  },
  
  // Deprecated (kept for backward compatibility)
  "organization_id": "org_abc123",  // DEPRECATED: use organization pointer
  "namespace_id": "ns_org_abc123_production",  // DEPRECATED: use namespace pointer
  
  // ACL
  "_acl": {
    "86cRDG7c4z": {"read": true, "write": true}
  }
}
```

### 6. Memory Class (Updated)

**Purpose**: Memory items with multi-tenant support

```javascript
{
  "objectId": "mem_abc123",
  "memoryId": "M3GCzpM9vP",
  "_created_at": ISODate("2025-01-01T00:00:00Z"),
  "_updated_at": ISODate("2025-01-01T00:00:00Z"),
  
  // Content
  "content": "Meeting notes...",
  "title": "Q4 Planning",
  "type": "TextMemoryItem",
  
  // User/Workspace (existing)
  "user": {
    "__type": "Pointer",
    "className": "_User",
    "objectId": "user_xyz789"
  },
  "workspace": {
    "__type": "Pointer",
    "className": "WorkSpace",
    "objectId": "workspace_123"
  },
  "developerUser": {
    "__type": "Pointer",
    "className": "DeveloperUser",
    "objectId": "devuser_123"
  },
  
  // NEW: Multi-tenant with Parse Pointers
  "organization": {
    "__type": "Pointer",
    "className": "Organization",
    "objectId": "org_abc123"
  },
  "namespace": {
    "__type": "Pointer",
    "className": "Namespace",
    "objectId": "ns_org_abc123_production"
  },
  
  // Deprecated (kept for backward compatibility)
  "organization_id": "org_abc123",  // DEPRECATED: use organization pointer
  "namespace_id": "ns_org_abc123_production",  // DEPRECATED: use namespace pointer
  
  // ACL
  "_acl": {
    "user_xyz789": {"read": true, "write": true},
    "org_abc123": {"read": true}
  },
  "_rperm": ["user_xyz789", "org_abc123"],
  "_wperm": ["user_xyz789"]
}
```

**Critical Indexes for Multi-Tenant Queries**:
```javascript
// Multi-tenant compound index (most important!)
db.Memory.createIndex({
  "_p_organization": 1,
  "_p_namespace": 1,
  "_created_at": -1
})

// Developer dashboard queries
db.Memory.createIndex({"_p_organization": 1, "_created_at": -1})

// Namespace-specific queries
db.Memory.createIndex({"_p_namespace": 1, "_created_at": -1})

// User-specific queries (existing)
db.Memory.createIndex({"_p_user": 1, "_created_at": -1})
```

## Parse Server Pointer Format

### How Pointers Are Stored in MongoDB

When you create a pointer in Parse Server:

**API Input (REST/SDK)**:
```javascript
{
  "organization": {
    "__type": "Pointer",
    "className": "Organization",
    "objectId": "org_abc123"
  }
}
```

**MongoDB Storage** (Parse Server automatically converts):
```javascript
{
  "_p_organization": "Organization$org_abc123"
}
```

**Querying Pointers**:
```javascript
// Via Parse SDK
query.equalTo("organization", organizationPointer);

// Via MongoDB directly
db.Memory.find({"_p_organization": "Organization$org_abc123"})
```

## Parse Server Relations

Relations are many-to-many relationships handled specially by Parse Server.

### Example: Organization team_members Relation

**Creating the relation via Parse REST API**:
```bash
# Add user to organization's team_members relation
curl -X PUT \
  -H "X-Parse-Application-Id: ${APPLICATION_ID}" \
  -H "X-Parse-REST-API-Key: ${REST_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "team_members": {
      "__op": "AddRelation",
      "objects": [
        {"__type": "Pointer", "className": "_User", "objectId": "user123"}
      ]
    }
  }' \
  https://yourserver.com/parse/classes/Organization/org_abc123
```

**Querying relations**:
```bash
# Get all team members for an organization
curl -X GET \
  -H "X-Parse-Application-Id: ${APPLICATION_ID}" \
  -H "X-Parse-REST-API-Key: ${REST_API_KEY}" \
  -G \
  --data-urlencode 'where={"$relatedTo":{"object":{"__type":"Pointer","className":"Organization","objectId":"org_abc123"},"key":"team_members"}}' \
  https://yourserver.com/parse/classes/_User
```

## Backward Compatibility Strategy

### Chat App (Existing Users)

**Before Migration**:
```javascript
_User {
  objectId: "user123",
  workspace: Pointer<WorkSpace>
}

Memory {
  user: Pointer<_User>,
  workspace: Pointer<WorkSpace>
}
```

**After Migration** (still works!):
```javascript
_User {
  objectId: "user123",
  workspace: Pointer<WorkSpace>,  // Still works
  user_type: null  // Not a developer
}

Memory {
  user: Pointer<_User>,
  workspace: Pointer<WorkSpace>,  // Still works
  organization: null,  // Not multi-tenant yet
  namespace: null
}
```

### Developer Dashboard (New System)

**For Developers**:
```javascript
_User {
  objectId: "dev_user",
  user_type: "DEVELOPER",
  organization: Pointer<Organization>,  // NEW
  workspace: Pointer<WorkSpace>  // Link maintained
}

Organization {
  owner: Pointer<_User:dev_user>,
  workspace: Pointer<WorkSpace>,  // Backward compat link
  team_members: Relation<_User>
}
```

### Query Patterns

**Chat App Query** (unchanged):
```javascript
// Works as before!
db.Memory.find({
  "_p_workspace": "WorkSpace$workspace123"
})
```

**Developer Dashboard Query** (new):
```javascript
// Get all memories for an organization
db.Memory.find({
  "_p_organization": "Organization$org_abc123"
})

// Get memories for a specific namespace
db.Memory.find({
  "_p_organization": "Organization$org_abc123",
  "_p_namespace": "Namespace$ns_org_abc123_production"
})
```

## Migration Strategy

1. **Phase 1**: Add new classes (Organization, Namespace, APIKey)
2. **Phase 2**: Discover workspace members via workspace_follower
3. **Phase 3**: Create Organizations with proper pointers
4. **Phase 4**: Create Relations (team_members, allowed_namespaces)
5. **Phase 5**: Backfill Memory pointers (optional, can be lazy)

See `scripts/migrate_to_multi_tenant_v3.py` for implementation.

## API Usage Examples

### Developer Gets All Their Memories

```python
# Via API (with organization context)
GET /api/v1/memories?organization_id=org_abc123

# Parse Query
query = Parse.Query("Memory")
query.equal_to("organization", organization_pointer)
memories = await query.find()
```

### Developer Gets Memories for Specific Namespace

```python
# Via API
GET /api/v1/memories?namespace_id=ns_org_abc123_production

# Parse Query
query = Parse.Query("Memory")
query.equal_to("namespace", namespace_pointer)
memories = await query.find()
```

### End User Gets Their Memories

```python
# Via API (with external_user_id)
GET /api/v1/memories?external_user_id=customer_123

# Parse Query
query = Parse.Query("Memory")
dev_user_query = Parse.Query("DeveloperUser")
dev_user_query.equal_to("external_id", "customer_123")
dev_user = await dev_user_query.first()

query.equal_to("developerUser", dev_user)
memories = await query.find()
```

## Summary

✅ **Parse Server native** - Uses proper Pointers and Relations  
✅ **Backward compatible** - Workspace and existing code still works  
✅ **Scalable** - Proper indexing for multi-tenant queries  
✅ **Flexible rate limits** - Org defaults + namespace overrides  
✅ **Industry-standard** - Based on MongoDB, Temporal, Mem0 best practices  
✅ **Role-based access** - Maps workspace roles to org roles  
✅ **Parse Dashboard visible** - Proper `_id`, `_acl`, `_rperm`, `_wperm`  

## References

- [Parse Server Data Types](https://docs.parseplatform.org/rest/guide/#data-types)
- [MongoDB Multi-Tenancy Guide](https://www.mongodb.com/docs/manual/tutorial/model-data-for-schema-design/)
- [Temporal Cloud Pricing & Limits](https://temporal.io/pricing)
- [Mem0 API Limits](https://docs.mem0.ai/)
- Papr Config: `config/cloud.yaml` - Rate limits per tier

