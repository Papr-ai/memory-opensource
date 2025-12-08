# Subscription Limits Implementation Guide

## Overview

This document describes the multi-tier subscription limit enforcement system implemented for Papr Memory, supporting both new developer platform tiers and legacy productivity app tiers.

## Architecture Changes

### 1. Multi-Level Count Tracking

Counts are now tracked at multiple levels in the hierarchy:

```
Organization (WorkSpace 1:1)
├── memoriesCount: Total memories across all team members
├── storageCount: Total storage in MB across all team members
│
├── workspace_followers (Individual team members)
│   ├── member1: memoriesCount, storageCount
│   └── member2: memoriesCount, storageCount
│
├── Namespace (Multi-tenant isolation)
│   ├── memoriesCount: Memories in this namespace
│   └── storageCount: Storage in this namespace
│
└── APIKey (Usage per API key)
    ├── memoriesCount: Memories created with this key
    ├── storageCount: Storage used by this key
    └── last_used_at: Timestamp of last usage
```

**Key Points:**
- **WorkSpace counts** = Organization-level aggregate (sum of all followers)
- **workspace_follower counts** = Individual user's contribution
- **Namespace counts** = Optional, for multi-tenant isolation
- **APIKey counts** = Optional, for tracking per-key usage

### 2. Configuration-Based Limits

Limits are now defined in `config/cloud.yaml` instead of hardcoded in the application:

```yaml
limits:
  # New Developer Platform Tiers
  developer:        # FREE tier
    max_memory_operations_per_month: 1000
    max_storage_gb: 1
    max_active_memories: 2500
    rate_limit_per_minute: 10
  
  starter:          # $100/mo
    max_memory_operations_per_month: 50000
    max_storage_gb: 10
    max_active_memories: 100000
    rate_limit_per_minute: 30
  
  growth:           # $500/mo
    max_memory_operations_per_month: 750000
    max_storage_gb: 100
    max_active_memories: 1000000
    rate_limit_per_minute: 100
  
  enterprise:       # Custom
    max_memory_operations_per_month: null  # Unlimited
    max_storage_gb: null
    max_active_memories: null
    rate_limit_per_minute: 500
  
  # Legacy Productivity App Tiers (for backward compatibility)
  free_trial:
    max_memory_operations_per_month: 1000
    max_storage_gb: 1
    max_active_memories: 2500
    rate_limit_per_minute: 10
  
  pro:
    max_memory_operations_per_month: 10000
    max_storage_gb: 5
    max_active_memories: 2500
    rate_limit_per_minute: 20
  
  business_plus:
    max_memory_operations_per_month: 100000
    max_storage_gb: 50
    max_active_memories: 20000
    rate_limit_per_minute: 50
```

### 3. Edition-Aware Enforcement

Limit checks are **only enabled in cloud edition**:

- **Open Source**: No limits enforced (self-hosted, unlimited)
- **Cloud Edition**: Full tier-based enforcement via Stripe

```python
from config.features import get_features
features = get_features()

if not features.is_cloud:
    # Skip all limit checks
    return None

if not features.is_enabled("subscription_enforcement"):
    # Skip if enforcement is disabled
    return None
```

## Implementation Details

### Updated Methods

#### `check_memory_limits()`

**Location**: `services/user_utils.py`

**New Parameters:**
```python
async def check_memory_limits(
    self, 
    increment_count: bool = True,
    memory_size_mb: float = 0.0,           # NEW: Memory size in MB
    namespace_id: Optional[str] = None,     # NEW: For namespace tracking
    api_key_id: Optional[str] = None        # NEW: For API key tracking
) -> Optional[Tuple[Dict[str, Any], int]]:
```

**What It Does:**
1. ✅ Checks cloud edition status (skips if open source)
2. ✅ Gets workspace and workspace_follower data
3. ✅ Increments counts at multiple levels:
   - `workspace_follower`: Individual user's counts
   - `WorkSpace`: Organization aggregate
   - `Namespace`: If namespace_id provided
   - `APIKey`: If api_key_id provided
4. ✅ Checks **BOTH** memory count AND storage limits
5. ✅ Returns detailed error with upgrade paths

**Example Error Response:**
```json
{
  "error": "Limit reached",
  "message": "You've reached the memory count (2500/2500) limit for your Developer plan. To continue adding memories, upgrade to Starter ($100/mo) or Growth ($500/mo) plan.\nVisit https://dashboard.papr.ai to manage your subscription.",
  "limits_exceeded": {
    "memory_count": true,
    "storage": false
  },
  "current": {
    "memory_count": 2500,
    "storage_gb": 0.85
  },
  "limits": {
    "memory_count": 2500,
    "storage_gb": 1
  },
  "tier": "developer",
  "is_trial": false
}
```

#### `check_interaction_limits()`

**Location**: `services/user_utils.py`

**Changes:**
- ✅ Now conditional on cloud edition
- ✅ Skips entirely in open source
- ✅ Uses config-based limits (still hardcoded in method, needs migration to config)

#### `check_interaction_limits_fast()`

**Location**: `services/user_utils.py`

**Changes:**
- ✅ Now conditional on cloud edition
- ✅ Optimized for <200ms response time
- ✅ Same logic as regular check

## Database Schema Updates

### Required Fields

Add these fields to your Parse Server classes:

#### **WorkSpace** (Organization level)
```javascript
{
  memoriesCount: Number,  // Total memories (sum of all followers)
  storageCount: Number,   // Total storage in MB (sum of all followers)
  // ... existing fields
}
```

#### **workspace_follower** (Individual level)
```javascript
{
  memoriesCount: Number,  // This user's memories
  storageCount: Number,   // This user's storage in MB
  // ... existing fields
}
```

#### **Namespace** (Multi-tenant)
```javascript
{
  objectId: String,
  name: String,
  organization: Pointer<Organization>,
  memoriesCount: Number,  // NEW
  storageCount: Number,   // NEW (in MB)
  // ... other fields
}
```

#### **APIKey** (Per-key tracking)
```javascript
{
  objectId: String,
  key: String,
  namespace: Pointer<Namespace>,
  organization: Pointer<Organization>,
  memoriesCount: Number,  // NEW
  storageCount: Number,   // NEW (in MB)
  last_used_at: Date,     // UPDATED on each use
  // ... other fields
}
```

## Usage Examples

### Adding Memory with Limits Check

```python
from services.user_utils import UserHelper

# Calculate memory size (example)
memory_content = "This is my memory content..."
memory_size_mb = len(memory_content.encode('utf-8')) / (1024 * 1024)

# Check limits before adding
user = UserHelper(user_id="user123", session_token="xyz")
limit_result = await user.check_memory_limits(
    increment_count=True,
    memory_size_mb=memory_size_mb,
    namespace_id="ns_prod_001",  # Optional
    api_key_id="key_abc123"      # Optional
)

if limit_result:
    # Limit exceeded
    error_response, status_code = limit_result
    return JSONResponse(content=error_response, status_code=status_code)

# Proceed with adding memory
# ...
```

### Checking Limits Without Incrementing

```python
# Check limits without modifying counts
limit_result = await user.check_memory_limits(
    increment_count=False  # Just check, don't increment
)

if limit_result:
    # Show warning to user
    pass
```

## Migration Strategy

### Phase 1: ✅ COMPLETED
- [x] Add legacy tier limits to config
- [x] Make checks conditional on cloud edition
- [x] Update `check_memory_limits()` to use config
- [x] Add storage size checking
- [x] Add workspace-level aggregation
- [x] Add namespace/API key tracking

### Phase 2: TODO
- [ ] Add `memoriesCount` and `storageCount` to WorkSpace class in Parse
- [ ] Add `memoriesCount` and `storageCount` to Namespace class in Parse
- [ ] Add `memoriesCount`, `storageCount`, `last_used_at` to APIKey class in Parse
- [ ] Backfill existing workspace counts (sum from followers)
- [ ] Test with real data

### Phase 3: TODO
- [ ] Update all API routes to pass `memory_size_mb` to `check_memory_limits()`
- [ ] Update routes to pass `namespace_id` and `api_key_id` when available
- [ ] Migrate interaction limits to config (currently still hardcoded)
- [ ] Add dashboard UI to show usage vs limits

### Phase 4: TODO
- [ ] Implement Parse Server Cloud Code hooks for automatic aggregation
- [ ] Add background job to sync counts (for consistency)
- [ ] Add alerts for users approaching limits (80%, 95%)
- [ ] Implement graceful degradation for limit errors

## Testing

### Local Development (Open Source Mode)

```bash
# Set environment to open source
export PAPR_EDITION=opensource

# Limits should NOT be enforced
poetry run python -m pytest tests/test_memory_limits.py -v

# Expected: All tests pass without limit errors
```

### Cloud Mode Testing

```bash
# Set environment to cloud
export PAPR_EDITION=cloud

# Test with developer tier
poetry run python -m pytest tests/test_memory_limits.py::test_developer_tier_limits -v

# Test storage limits
poetry run python -m pytest tests/test_memory_limits.py::test_storage_limits -v

# Test namespace tracking
poetry run python -m pytest tests/test_memory_limits.py::test_namespace_tracking -v
```

### Manual Testing

```bash
# 1. Create test user with developer tier
# 2. Add 2500 memories (should succeed)
# 3. Try to add 2501st memory (should fail with limit error)
# 4. Check error message mentions correct tier and upgrade path
# 5. Verify counts are tracked at workspace level
```

## Monitoring & Observability

### Key Metrics to Track

1. **Limit Exceeded Events** (by tier)
   - Count of 403 responses from `check_memory_limits()`
   - Group by `tier`, `limit_type` (memory_count vs storage)

2. **Upgrade Conversions**
   - Track users who upgrade after hitting limits
   - Measure time between limit hit and upgrade

3. **Usage by Tier**
   - Average memories per user by tier
   - Average storage per user by tier
   - Percentage of limit used by tier

4. **Namespace Usage** (for developer platform)
   - Memories per namespace
   - Storage per namespace
   - API key usage distribution

### Example Amplitude Events

```python
# Track limit exceeded
amplitude.track({
    'event_type': 'memory_limit_exceeded',
    'user_id': user_id,
    'event_properties': {
        'tier': 'developer',
        'limit_type': 'memory_count',
        'current_count': 2500,
        'limit': 2500,
        'overage_amount': 1
    }
})

# Track successful upgrade
amplitude.track({
    'event_type': 'subscription_upgraded',
    'user_id': user_id,
    'event_properties': {
        'from_tier': 'developer',
        'to_tier': 'starter',
        'triggered_by': 'limit_exceeded'
    }
})
```

## Parse Server Implementation (Alternative Approach)

If you decide to implement count aggregation in Parse Server Cloud Code (cleaner but requires Parse Server changes):

```javascript
// cloud/main.js

// Automatically aggregate workspace counts when workspace_follower changes
Parse.Cloud.afterSave("workspace_follower", async (request) => {
  const follower = request.object;
  const workspace = follower.get("workspace");
  
  if (!workspace) return;
  
  // Query all followers for this workspace
  const query = new Parse.Query("workspace_follower");
  query.equalTo("workspace", workspace);
  
  const followers = await query.find({ useMasterKey: true });
  
  // Sum up counts
  let totalMemories = 0;
  let totalStorage = 0;
  
  followers.forEach(f => {
    totalMemories += f.get("memoriesCount") || 0;
    totalStorage += f.get("storageCount") || 0;
  });
  
  // Update workspace
  workspace.set("memoriesCount", totalMemories);
  workspace.set("storageCount", totalStorage);
  await workspace.save(null, { useMasterKey: true });
});
```

## FAQs

### Q: Why track at workspace level instead of workspace_follower?

**A:** Organization-level limits apply to the entire team, not individual members. If a team of 5 developers has a 100K memory limit, it's shared across all members. Individual tracking helps attribution, but enforcement is at the org level.

### Q: What happens to counts when a memory is deleted?

**A:** You need to implement a `check_memory_deletion()` method that decrements counts at all levels (follower, workspace, namespace, API key).

### Q: How do we handle race conditions in count updates?

**A:** Use Parse Server's atomic increment operations:
```python
{
    "memoriesCount": {"__op": "Increment", "amount": 1}
}
```

### Q: Should we enforce limits on search/retrieval too?

**A:** Different approach - use rate limiting (requests per minute) instead of counting operations. Memory operations (add/update/delete) count against monthly quota, but reads are rate-limited only.

### Q: What about batch operations?

**A:** Calculate total size of batch, check limits BEFORE processing, then increment by batch size (not individual items).

## Next Steps

1. **Database Migration**: Add count fields to WorkSpace, Namespace, APIKey
2. **Backfill Counts**: Run script to calculate existing counts
3. **Route Updates**: Pass `memory_size_mb`, `namespace_id`, `api_key_id` to limit checks
4. **Dashboard**: Build UI to show usage vs limits
5. **Alerts**: Implement proactive notifications at 80% and 95% usage

## Related Documents

- **Rate Limits**: See `docs/RATE_LIMITS_REFERENCE.md`
- **Multi-Tenant Schema**: See `docs/MULTI_TENANT_SCHEMA_DESIGN.md`
- **Feature Flags**: See `config/features.py`
- **Cloud Config**: See `config/cloud.yaml`

---

**Last Updated:** 2025-10-03  
**Author:** Implementation based on requirements  
**Status:** Phase 1 Complete, Phase 2-4 Pending

