# Check Memory Limits - Organization ID Update

## Summary

Updated `check_memory_limits` to accept `organization_id` as an input parameter, allowing it to be passed directly from the auth_response's multi-tenant resolution instead of extracting it from the subscription.

## Changes Made

### 1. Updated `services/user_utils.py`

**Modified `check_memory_limits` method signature:**

```python
async def check_memory_limits(
    self,
    increment_count: bool = True,
    memory_size_mb: float = 0.0,
    namespace_id: Optional[str] = None,
    api_key_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    organization_id: Optional[str] = None  # NEW PARAMETER
) -> Optional[Tuple[Dict[str, Any], int]]:
```

**Updated organization count logic:**

Instead of always extracting `organization_id` from the subscription, the method now:
1. Uses the provided `organization_id` parameter if available
2. Falls back to extracting from subscription if not provided
3. Logs which source was used for better debugging

```python
# Use provided organization_id or fall back to extracting from subscription
org_id_to_use = organization_id
if not org_id_to_use:
    # Fallback: Get organization_id from workspace if it has a subscription with organization link
    org_id_to_use = subscription.get('organization', {}).get('objectId') if isinstance(subscription.get('organization'), dict) else None
    logger.info(f"organization_id extracted from subscription: {org_id_to_use}")
else:
    logger.info(f"Using provided organization_id: {org_id_to_use}")
```

### 2. Updated `services/memory_service.py`

**Modified `handle_incoming_memory` function:**

Updated the call to `check_memory_limits` to extract and pass `organization_id` and `namespace_id` from the `memory_request`:

```python
# Extract organization_id and namespace_id from memory_request if available
organization_id = memory_request.organization_id if memory_request and hasattr(memory_request, 'organization_id') else None
namespace_id = memory_request.namespace_id if memory_request and hasattr(memory_request, 'namespace_id') else None

limit_check = await developer_user.check_memory_limits(
    workspace_id=workspace_id,
    organization_id=organization_id,  # NEW
    namespace_id=namespace_id         # Now passed explicitly
)
```

## How It Works

### Single Memory Addition (`add_memory_v1`)

1. Auth response resolves organization/namespace IDs for multi-tenant
2. These IDs are set in `AddMemoryRequest.organization_id` and `AddMemoryRequest.namespace_id`
3. `common_add_memory_handler` calls `handle_incoming_memory`
4. `handle_incoming_memory` extracts these IDs from `memory_request`
5. Passes them to `check_memory_limits` which updates organization counts

### Batch Memory Addition (`add_memory_batch_v1`)

1. Auth response resolves organization/namespace IDs for multi-tenant
2. Batch-level IDs are set in `BatchMemoryRequest.organization_id` and `BatchMemoryRequest.namespace_id`
3. Each memory in the batch inherits these IDs (or has its own if specified)
4. `common_add_memory_batch_handler` processes each memory individually
5. For each memory, `handle_incoming_memory` is called
6. Same flow as single memory addition

## Benefits

### 1. **Cleaner Architecture**
- Organization ID is resolved once during auth
- Passed explicitly through the call chain
- No need to re-extract from subscription at multiple levels

### 2. **Multi-Tenant Support**
- Organization ID comes from auth_response which has proper multi-tenant resolution
- Supports both legacy (user-based) and organization-based authentication
- Namespace ID is also properly tracked

### 3. **Better Logging**
- Clear logs show whether organization_id was provided or extracted
- Easier to debug multi-tenant issues
- Can track which organization each memory belongs to

### 4. **Backward Compatible**
- Falls back to extracting from subscription if organization_id not provided
- Existing code that doesn't pass organization_id continues to work
- Gradual migration path

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                  Auth Response (Multi-Tenant)                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ organization_id: "org_abc123"                             │  │
│  │ namespace_id: "ns_xyz789"                                 │  │
│  │ workspace_id: "workspace_456"                             │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              AddMemoryRequest / BatchMemoryRequest               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ organization_id: "org_abc123"  (from auth_response)      │  │
│  │ namespace_id: "ns_xyz789"      (from auth_response)      │  │
│  │ content: "..."                                            │  │
│  │ metadata: {...}                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    handle_incoming_memory                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Extracts:                                                 │  │
│  │   organization_id = memory_request.organization_id        │  │
│  │   namespace_id = memory_request.namespace_id              │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     check_memory_limits                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Uses organization_id to update:                           │  │
│  │   - Organization.memoriesCount                            │  │
│  │   - Organization.storageCount                             │  │
│  │                                                            │  │
│  │ Uses namespace_id to update:                              │  │
│  │   - Namespace.memoriesCount                               │  │
│  │   - Namespace.storageCount                                │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Schema Fields

### AddMemoryRequest
```python
class AddMemoryRequest:
    content: str
    type: MemoryType
    metadata: Optional[MemoryMetadata]
    organization_id: Optional[str]  # Used for organization count tracking
    namespace_id: Optional[str]     # Used for namespace count tracking
```

### BatchMemoryRequest
```python
class BatchMemoryRequest:
    memories: List[AddMemoryRequest]
    organization_id: Optional[str]  # Applied to all memories if not set individually
    namespace_id: Optional[str]     # Applied to all memories if not set individually
    batch_size: Optional[int]
    webhook_url: Optional[str]
```

## Database Updates

When a memory is added with `organization_id` and `namespace_id`:

1. **WorkSpace counts updated:**
   - `memoriesCount` += 1
   - `storageCount` += memory_size_mb

2. **Organization counts updated:**
   - `memoriesCount` = workspace_memories_count (synced)
   - `storageCount` = workspace_storage_mb (synced)

3. **Namespace counts updated:**
   - `memoriesCount` += 1 (incremental)
   - `storageCount` += memory_size_mb (incremental)

4. **API Key counts updated (if api_key_id provided):**
   - `memoriesCount` += 1
   - `storageCount` += memory_size_mb
   - `last_used_at` = now()

## Testing

### Test Single Memory with Organization ID

```python
response = await client.post(
    "/v1/memory",
    headers={"Authorization": f"APIKey {org_api_key}"},
    json={
        "content": "Test memory",
        "organization_id": "org_abc123",
        "namespace_id": "ns_xyz789"
    }
)

# Verify counts were updated
org = await get_organization("org_abc123")
assert org["memoriesCount"] > 0

namespace = await get_namespace("ns_xyz789")
assert namespace["memoriesCount"] > 0
```

### Test Batch Memories with Organization ID

```python
response = await client.post(
    "/v1/memories/batch",
    headers={"Authorization": f"APIKey {org_api_key}"},
    json={
        "organization_id": "org_abc123",
        "namespace_id": "ns_xyz789",
        "memories": [
            {"content": "Memory 1"},
            {"content": "Memory 2"},
            {"content": "Memory 3"}
        ]
    }
)

# Verify counts were updated for all 3 memories
org = await get_organization("org_abc123")
assert org["memoriesCount"] >= 3
```

## Migration Notes

### Existing Code
Existing code that doesn't pass `organization_id` will continue to work:

```python
# Still works - extracts organization_id from subscription
limit_check = await user.check_memory_limits(
    workspace_id=workspace_id
)
```

### New Code
New code should pass organization_id explicitly:

```python
# Recommended - passes organization_id directly
limit_check = await user.check_memory_limits(
    workspace_id=workspace_id,
    organization_id=memory_request.organization_id,
    namespace_id=memory_request.namespace_id
)
```

## Next Steps

1. ✅ Updated `check_memory_limits` to accept `organization_id` parameter
2. ✅ Updated `handle_incoming_memory` to pass organization_id and namespace_id
3. ✅ Added fallback logic for backward compatibility
4. ✅ Improved logging for debugging
5. Consider updating other places that call `check_memory_limits` to pass organization_id explicitly

## Related Documentation

- `docs/multi-tenant/ORGANIZATION_BASED_AUTH.md` - Multi-tenant authentication
- `docs/SUBSCRIPTION_LIMITS_IMPLEMENTATION.md` - Subscription limits
- `docs/API_KEY_LAST_USED_UPDATE.md` - API key tracking

