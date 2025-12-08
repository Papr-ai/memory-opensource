# Interaction Pointers Update

## Summary

Updated the Interaction tracking system to include pointers to Organization, Namespace, and APIKey classes. This enables better tracking and analytics for multi-tenant usage and API key management.

## Changes Made

### 1. ✅ Updated `check_interaction_limits_fast` Signature

Added `organization_id` and `namespace_id` parameters:

```python
async def check_interaction_limits_fast(
    self, 
    interaction_type: str = 'mini',
    memory_graph = None,
    operation: Optional['MemoryOperationType'] = None,
    batch_size: Optional[int] = None,
    enable_agentic_graph: bool = False,
    enable_rank_results: bool = False,
    api_key_id: Optional[str] = None,              # Existing
    organization_id: Optional[str] = None,         # NEW
    namespace_id: Optional[str] = None             # NEW
) -> Optional[Tuple[Dict[str, Any], int, bool]]:
```

### 2. ✅ Updated Interaction Count Methods

**Modified all interaction tracking methods to accept and pass the new fields:**

- `_update_interaction_count_fast()`
- `_update_interaction_count_mongo()`
- `_update_interaction_count_parse()`

All now accept: `organization_id`, `namespace_id`, and `api_key_id`

### 3. ✅ Updated MongoDB Interaction Updates

When updating interactions in MongoDB, now sets pointers:

```python
# Build update data
update_data = {
    "count": current_count,
    "operation_type": operation_type,
    "_updated_at": datetime.now()
}

# Add pointers if provided
if organization_id:
    update_data["_p_organization"] = f"Organization${organization_id}"
if namespace_id:
    update_data["_p_namespace"] = f"Namespace${namespace_id}"
if api_key_id:
    update_data["_p_apiKey"] = f"APIKey${api_key_id}"
```

### 4. ✅ Updated Parse Server Interaction Creation

When creating new interactions via Parse Server:

```python
new_interaction = {
    "workspace": {"__type": "Pointer", "className": "WorkSpace", "objectId": workspace_id},
    "user": {"__type": "Pointer", "className": "_User", "objectId": self.id},
    "type": interaction_type,
    "month": current_month,
    "year": current_year,
    "count": increment_by,
    "operation_type": operation_type
}

# Add pointers if provided
if company_id:
    new_interaction["company"] = {"__type": "Pointer", "className": "Company", "objectId": company_id}
if subscription_id:
    new_interaction["subscription"] = {"__type": "Pointer", "className": "Subscription", "objectId": subscription_id}
if organization_id:
    new_interaction["organization"] = {"__type": "Pointer", "className": "Organization", "objectId": organization_id}  # NEW
if namespace_id:
    new_interaction["namespace"] = {"__type": "Pointer", "className": "Namespace", "objectId": namespace_id}  # NEW
if api_key_id:
    new_interaction["apiKey"] = {"__type": "Pointer", "className": "APIKey", "objectId": api_key_id}  # NEW
```

### 5. ✅ Updated Parse Server Interaction Updates

When updating existing interactions via Parse Server:

```python
update_data = {
    "count": current_count,
    "operation_type": operation_type
}

# Add pointers if provided
if organization_id:
    update_data["organization"] = {"__type": "Pointer", "className": "Organization", "objectId": organization_id}
if namespace_id:
    update_data["namespace"] = {"__type": "Pointer", "className": "Namespace", "objectId": namespace_id}
if api_key_id:
    update_data["apiKey"] = {"__type": "Pointer", "className": "APIKey", "objectId": api_key_id}
```

## Updated Interaction Schema

The Interaction class now includes:

```json
{
  "workspace": {
    "__type": "Pointer",
    "className": "WorkSpace",
    "objectId": "HfXpufNZ7m"
  },
  "user": {
    "__type": "Pointer",
    "className": "_User",
    "objectId": "86cRDG7c4z"
  },
  "type": "mini",
  "month": 11,
  "year": 2025,
  "count": 9,
  "operation_type": "add_memory_v1",
  "subscription": {
    "__type": "Pointer",
    "className": "Subscription",
    "objectId": "PnNWXv4hib"
  },
  "organization": {                              // NEW
    "__type": "Pointer",
    "className": "Organization",
    "objectId": "org_abc123"
  },
  "namespace": {                                 // NEW
    "__type": "Pointer",
    "className": "Namespace",
    "objectId": "ns_xyz789"
  },
  "apiKey": {                                    // NEW
    "__type": "Pointer",
    "className": "APIKey",
    "objectId": "key_def456"
  },
  "createdAt": "2025-11-10T18:27:00.489Z",
  "updatedAt": "2025-11-10T18:40:53.920Z",
  "objectId": "ymiZjcTHFa"
}
```

## Benefits

### 1. **Organization-Level Analytics**
Track interaction usage per organization:

```javascript
// Get all interactions for an organization
Parse.Query("Interaction")
  .equalTo("organization", {
    __type: "Pointer",
    className: "Organization",
    objectId: "org_abc123"
  })
  .find();
```

### 2. **Namespace-Level Analytics**
Track interaction usage per namespace for multi-tenant isolation:

```javascript
// Get all interactions for a namespace
Parse.Query("Interaction")
  .equalTo("namespace", {
    __type: "Pointer",
    className: "Namespace",
    objectId: "ns_xyz789"
  })
  .find();
```

### 3. **API Key-Level Analytics**
Track which API keys are consuming interactions:

```javascript
// Get all interactions for an API key
Parse.Query("Interaction")
  .equalTo("apiKey", {
    __type: "Pointer",
    className: "APIKey",
    objectId: "key_def456"
  })
  .find();
```

### 4. **Comprehensive Dashboards**
Enable dashboards showing:
- Organization interaction usage over time
- Namespace interaction consumption patterns
- API key usage breakdown
- Operation-type distribution per organization/namespace/API key

## Usage Flow

```
Route Handler (e.g., add_memory_v1)
  └─> Extracts organization_id, namespace_id from auth_response
      └─> Calls check_interaction_limits_fast(
            organization_id=organization_id,
            namespace_id=namespace_id,
            api_key_id=api_key_id
          )
          └─> Updates interaction with all pointers
              └─> Interaction record created/updated with:
                  - workspace pointer
                  - user pointer
                  - subscription pointer
                  - organization pointer (NEW)
                  - namespace pointer (NEW)
                  - apiKey pointer (NEW)
                  - operation_type (e.g., "add_memory_v1")
```

## Next Steps

### 1. Update Route Handlers

All route handlers that call `check_interaction_limits_fast` need to pass `organization_id` and `namespace_id`:

```python
# Extract from auth_response
organization_id = auth_response.organization_id if hasattr(auth_response, 'organization_id') else None
namespace_id = auth_response.namespace_id if hasattr(auth_response, 'namespace_id') else None

# Pass to check_interaction_limits_fast
limit_check = await user_instance.check_interaction_limits_fast(
    interaction_type='mini',
    memory_graph=memory_graph,
    operation=MemoryOperationType.ADD_MEMORY,
    api_key_id=api_key_id,
    organization_id=organization_id,  # NEW
    namespace_id=namespace_id          # NEW
)
```

### 2. Update Parse Server Schema

Ensure the Interaction class in Parse Server has columns for:
- `organization` (Pointer to Organization)
- `namespace` (Pointer to Namespace)  
- `apiKey` (Pointer to APIKey)

These will be auto-created when the first interaction with these fields is saved, but you may want to add them manually with proper indexes.

### 3. Add Indexes for Performance

Create Parse Server indexes:

```javascript
// In Parse Dashboard or cloud code
db.Interaction.createIndex({ "_p_organization": 1 });
db.Interaction.createIndex({ "_p_namespace": 1 });
db.Interaction.createIndex({ "_p_apiKey": 1 });
db.Interaction.createIndex({ "_p_organization": 1, "month": 1, "year": 1 });
db.Interaction.createIndex({ "_p_namespace": 1, "month": 1, "year": 1 });
db.Interaction.createIndex({ "_p_apiKey": 1, "month": 1, "year": 1 });
```

### 4. Create Dashboard Queries

Add Cloud Functions for aggregating interaction data:

```javascript
// cloud/main.js
Parse.Cloud.define("getOrganizationInteractions", async (request) => {
  const { organizationId, month, year } = request.params;
  
  const query = new Parse.Query("Interaction");
  query.equalTo("organization", {
    __type: "Pointer",
    className: "Organization",
    objectId: organizationId
  });
  query.equalTo("month", month);
  query.equalTo("year", year);
  
  const interactions = await query.find({ useMasterKey: true });
  
  // Aggregate by operation_type
  const breakdown = {};
  let totalCount = 0;
  
  interactions.forEach(interaction => {
    const opType = interaction.get("operation_type") || "unknown";
    const count = interaction.get("count") || 0;
    breakdown[opType] = (breakdown[opType] || 0) + count;
    totalCount += count;
  });
  
  return { total: totalCount, breakdown };
});
```

### 5. Test Complete Flow

```python
# Test that all fields are being tracked
response = await client.post(
    "/v1/memory",
    headers={"Authorization": f"APIKey {org_api_key}"},
    json={
        "content": "Test memory",
        "organization_id": "org_abc123",
        "namespace_id": "ns_xyz789"
    }
)

# Verify interaction was created with all pointers
interaction = await Parse.Query("Interaction").descending("createdAt").first()
assert interaction.get("organization").id == "org_abc123"
assert interaction.get("namespace").id == "ns_xyz789"
assert interaction.get("apiKey") is not None
assert interaction.get("operation_type") == "add_memory_v1"
```

## Related Documentation

- `docs/OPERATION_TRACKING.md` - Operation type tracking
- `docs/API_KEY_LAST_USED_UPDATE.md` - API key tracking
- `docs/CHECK_MEMORY_LIMITS_ORGANIZATION_UPDATE.md` - Organization tracking in memory limits
- `docs/COMPLETE_PARAMETER_UPDATE_SUMMARY.md` - Complete parameter updates

## Status

✅ **COMPLETE** - All interaction tracking methods now include organization, namespace, and apiKey pointers!

---

**No Linter Errors** ✨

