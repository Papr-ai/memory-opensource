# Subscription Limits Implementation - Summary

## What Was Implemented

### ✅ 1. Configuration-Based Limits

**File**: `config/cloud.yaml`

Added support for **both new and legacy tiers**:

#### New Developer Platform Tiers
- `developer` (FREE): 1K operations, 1GB storage, 2.5K memories
- `starter` ($100/mo): 50K operations, 10GB storage, 100K memories
- `growth` ($500/mo): 750K operations, 100GB storage, 1M memories
- `enterprise` (Custom): Unlimited

#### Legacy Productivity App Tiers (Backward Compatible)
- `free_trial`: Same as developer
- `pro`: 10K operations, 5GB storage, 2.5K memories
- `business_plus`: 100K operations, 50GB storage, 20K memories

### ✅ 2. Cloud Edition Conditional Checks

**Files**: `services/user_utils.py`

All limit checks now respect the edition:
- **Open Source**: NO limits enforced (self-hosted)
- **Cloud**: Full tier-based enforcement

```python
from config.features import get_features
features = get_features()

if not features.is_cloud:
    return None  # Skip all checks
```

Applied to:
- ✅ `check_memory_limits()`
- ✅ `check_interaction_limits()`
- ✅ `check_interaction_limits_fast()`

### ✅ 3. Dual Limit Checking (Count + Storage)

**File**: `services/user_utils.py` - `check_memory_limits()`

Now checks **BOTH**:
1. **Memory Count**: Number of active memories
2. **Storage Size**: Total storage in GB

Example error:
```
"You've reached the memory count (2500/2500) and storage (1.02GB/1GB) limit..."
```

### ✅ 4. Multi-Level Count Tracking

**File**: `services/user_utils.py` - `check_memory_limits()`

Counts are now incremented at **4 levels**:

```
1. workspace_follower (individual user)
   ├── memoriesCount += 1
   └── storageCount += memory_size_mb

2. WorkSpace (organization aggregate)
   ├── memoriesCount += 1
   └── storageCount += memory_size_mb

3. Namespace (if namespace_id provided)
   ├── memoriesCount += 1
   └── storageCount += memory_size_mb

4. APIKey (if api_key_id provided)
   ├── memoriesCount += 1
   ├── storageCount += memory_size_mb
   └── last_used_at = now()
```

### ✅ 5. Enhanced Error Messages

**File**: `services/user_utils.py`

Error responses now include:
- Which limit was exceeded (count, storage, or both)
- Current usage vs limit for both metrics
- Tier-specific upgrade paths
- Different messages for developer vs productivity app tiers

Example response:
```json
{
  "error": "Limit reached",
  "message": "You've reached the memory count (2500/2500) limit...",
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

## Files Modified

1. ✅ `config/cloud.yaml` - Added legacy tiers
2. ✅ `services/user_utils.py` - Updated 3 methods
3. ✅ `docs/SUBSCRIPTION_LIMITS_IMPLEMENTATION.md` - Comprehensive guide
4. ✅ `SUBSCRIPTION_LIMITS_UPDATE_SUMMARY.md` - This file

## Database Schema Changes Required

⚠️ **ACTION NEEDED**: Add these fields to Parse Server classes:

### WorkSpace
```javascript
{
  memoriesCount: Number,  // NEW: Total for organization
  storageCount: Number    // NEW: Total in MB
}
```

### Namespace
```javascript
{
  memoriesCount: Number,  // NEW
  storageCount: Number    // NEW: In MB
}
```

### APIKey
```javascript
{
  memoriesCount: Number,  // NEW
  storageCount: Number,   // NEW: In MB
  last_used_at: Date      // UPDATED on each use
}
```

## What Still Needs To Be Done

### Phase 2: Database Setup
- [ ] Add `memoriesCount` and `storageCount` fields to `WorkSpace` class
- [ ] Add `memoriesCount` and `storageCount` fields to `Namespace` class
- [ ] Add `memoriesCount`, `storageCount`, `last_used_at` fields to `APIKey` class
- [ ] Run backfill script to populate `WorkSpace.memoriesCount` and `storageCount` from existing `workspace_follower` records

### Phase 3: API Route Updates
- [ ] Update all memory creation routes to calculate `memory_size_mb`
- [ ] Pass `memory_size_mb` to `check_memory_limits()`
- [ ] Extract `namespace_id` from request (if multi-tenant API is used)
- [ ] Extract `api_key_id` from request (if API key auth is used)
- [ ] Handle memory deletion (decrement counts)
- [ ] Handle memory updates (adjust storage size)

### Phase 4: Testing
- [ ] Write unit tests for `check_memory_limits()` with new parameters
- [ ] Test open source mode (should skip all checks)
- [ ] Test cloud mode with each tier
- [ ] Test storage limit enforcement
- [ ] Test memory count limit enforcement
- [ ] Test both limits exceeded simultaneously
- [ ] Test namespace tracking
- [ ] Test API key tracking

### Phase 5: Monitoring
- [ ] Add Amplitude events for limit exceeded
- [ ] Add dashboard to show usage vs limits
- [ ] Implement alerts at 80% and 95% usage
- [ ] Track conversion rates (limit hit → upgrade)

## How to Calculate Memory Size

### In API Routes

```python
# For text-based memories
memory_content = request_data.get("content", "")
memory_size_bytes = len(memory_content.encode('utf-8'))
memory_size_mb = memory_size_bytes / (1024 * 1024)

# For document uploads
file_size_bytes = len(file_content)
memory_size_mb = file_size_bytes / (1024 * 1024)

# Pass to limit check
limit_result = await user.check_memory_limits(
    increment_count=True,
    memory_size_mb=memory_size_mb,
    namespace_id=namespace_id,
    api_key_id=api_key_id
)
```

## How to Extract Namespace/API Key IDs

### From Request Headers

```python
# API Key authentication
api_key = request.headers.get("X-API-Key")
if api_key:
    # Look up API key in database to get api_key_id
    api_key_obj = await get_api_key_by_value(api_key)
    api_key_id = api_key_obj.get("objectId")
    namespace_id = api_key_obj.get("namespace", {}).get("objectId")
else:
    api_key_id = None
    namespace_id = None

# Pass to limit check
limit_result = await user.check_memory_limits(
    increment_count=True,
    memory_size_mb=memory_size_mb,
    namespace_id=namespace_id,
    api_key_id=api_key_id
)
```

## Backfill Script Example

```python
# scripts/backfill_workspace_counts.py

import asyncio
import httpx
from datetime import datetime

PARSE_SERVER_URL = "your-parse-url"
PARSE_APPLICATION_ID = "your-app-id"
PARSE_MASTER_KEY = "your-master-key"

async def backfill_workspace_counts():
    """
    Backfill memoriesCount and storageCount for all WorkSpace objects
    by aggregating from workspace_follower records.
    """
    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get all workspaces
        workspaces_url = f"{PARSE_SERVER_URL}/parse/classes/WorkSpace"
        workspaces_response = await client.get(workspaces_url, headers=headers)
        workspaces = workspaces_response.json().get("results", [])
        
        print(f"Found {len(workspaces)} workspaces")
        
        for workspace in workspaces:
            workspace_id = workspace["objectId"]
            
            # Get all followers for this workspace
            followers_url = f"{PARSE_SERVER_URL}/parse/classes/workspace_follower"
            params = {
                "where": json.dumps({
                    "workspace": {
                        "__type": "Pointer",
                        "className": "WorkSpace",
                        "objectId": workspace_id
                    }
                })
            }
            
            followers_response = await client.get(followers_url, headers=headers, params=params)
            followers = followers_response.json().get("results", [])
            
            # Sum up counts
            total_memories = sum(f.get("memoriesCount", 0) or 0 for f in followers)
            total_storage = sum(f.get("storageCount", 0) or 0 for f in followers)
            
            # Update workspace
            update_url = f"{PARSE_SERVER_URL}/parse/classes/WorkSpace/{workspace_id}"
            update_data = {
                "memoriesCount": total_memories,
                "storageCount": total_storage
            }
            
            update_response = await client.put(update_url, headers=headers, json=update_data)
            
            if update_response.status_code == 200:
                print(f"✓ Updated workspace {workspace_id}: {total_memories} memories, {total_storage}MB")
            else:
                print(f"✗ Failed to update workspace {workspace_id}: {update_response.text}")

if __name__ == "__main__":
    asyncio.run(backfill_workspace_counts())
```

## Testing the Implementation

### 1. Test Open Source Mode (No Limits)

```bash
export PAPR_EDITION=opensource
poetry run python -m pytest tests/ -k "memory_limits" -v

# Expected: All operations succeed without limit checks
```

### 2. Test Cloud Mode (With Limits)

```bash
export PAPR_EDITION=cloud
poetry run python scripts/test_limits.py

# Script should:
# 1. Create test user with developer tier
# 2. Add 2500 memories (should succeed)
# 3. Try to add 2501st (should fail)
# 4. Verify error message is correct
```

### 3. Test Storage Limits

```python
# Add memories with large content until storage limit hit
# Should get error: "storage (1.02GB/1GB) limit"
```

## Configuration Reference

### Cloud Edition (`config/cloud.yaml`)

```yaml
limits:
  developer:
    max_memory_operations_per_month: 1000
    max_storage_gb: 1
    max_active_memories: 2500
    rate_limit_per_minute: 10
  # ... other tiers
```

### Feature Flags (`config/features.py`)

```python
from config import get_features

features = get_features()

if features.is_cloud:
    # Enforce limits
    pass

if features.is_enabled("subscription_enforcement"):
    # Check subscription
    pass

limits = features.get_tier_limits("developer")
# Returns: {"max_active_memories": 2500, "max_storage_gb": 1, ...}
```

## Migration Checklist

- [x] Add legacy tiers to `config/cloud.yaml`
- [x] Update `check_memory_limits()` to be edition-aware
- [x] Update `check_memory_limits()` to check storage
- [x] Update `check_memory_limits()` to use config-based limits
- [x] Update `check_memory_limits()` to track at workspace level
- [x] Update `check_memory_limits()` to support namespace/API key tracking
- [x] Update `check_interaction_limits()` to be edition-aware
- [x] Update `check_interaction_limits_fast()` to be edition-aware
- [x] Create comprehensive documentation

- [ ] Add database fields to Parse Server classes
- [ ] Run backfill script for existing data
- [ ] Update API routes to pass memory_size_mb
- [ ] Update API routes to pass namespace_id/api_key_id
- [ ] Implement memory deletion count decrements
- [ ] Write tests for new functionality
- [ ] Deploy to staging and test end-to-end
- [ ] Monitor for issues and iterate

## Questions & Answers

**Q: What about parse server? Should we implement aggregation there?**

**A**: Two options:

1. **Python-only (current implementation)**: 
   - ✅ No Parse Server code changes needed
   - ✅ Works immediately
   - ⚠️ Requires manual count management in Python
   - ⚠️ Risk of count drift if updates fail

2. **Parse Server Cloud Code (recommended for production)**:
   - ✅ Automatic aggregation via afterSave hooks
   - ✅ Always consistent counts
   - ⚠️ Requires Parse Server deployment
   - ⚠️ More complex to debug

**Recommendation**: Start with Python-only (current implementation) to unblock development. Migrate to Parse Server Cloud Code hooks in Phase 5 for production-grade consistency.

---

**Status**: Phase 1 Complete ✅  
**Next Steps**: Add database fields and backfill existing data

