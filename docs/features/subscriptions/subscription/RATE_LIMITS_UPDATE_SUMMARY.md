# Rate Limits Update Summary

Updated rate limits across all configuration files and documentation to match Papr's actual pricing page.

## Changes Made

### 1. ✅ Updated `config/cloud.yaml`

**Before** (incorrect):
```yaml
limits:
  starter:
    max_memories_per_month: 1000
    max_api_calls_per_day: 1000
  growth:
    max_memories_per_month: 10000
    max_api_calls_per_day: 10000
```

**After** (matches pricing page):
```yaml
limits:
  developer:  # FREE tier
    max_memory_operations_per_month: 1000
    max_storage_gb: 1
    max_active_memories: 2500
    rate_limit_per_minute: 10
  
  starter:  # $100/mo
    max_memory_operations_per_month: 50000
    max_storage_gb: 10
    max_active_memories: 100000
    rate_limit_per_minute: 30
  
  growth:  # $500/mo
    max_memory_operations_per_month: 750000
    max_storage_gb: 100
    max_active_memories: 1000000
    rate_limit_per_minute: 100
  
  enterprise:  # Custom
    max_memory_operations_per_month: null  # Unlimited
    max_storage_gb: null
    max_active_memories: null
    rate_limit_per_minute: 500
```

**Key Changes**:
- ✅ Renamed `max_memories_per_month` → `max_memory_operations_per_month`
- ✅ Removed `max_api_calls_per_day` (redundant with memory operations)
- ✅ Added `max_storage_gb` (from pricing page)
- ✅ Added `max_active_memories` (from pricing page)
- ✅ Added `developer` free tier
- ✅ Updated tier names: removed "pro", "business_plus"
- ✅ Changed default tier from "trial" → "developer"

### 2. ✅ Updated `models/parse_server.py`

**Organization Model**:
```python
# Before
plan_tier: str = Field(default="trial")
rate_limits: {
    "max_memories_per_month": 1000,
    "max_api_calls_per_day": 1000,
    "rate_limit_per_minute": 20
}

# After (Developer tier defaults)
plan_tier: str = Field(default="developer")
rate_limits: {
    "max_memory_operations_per_month": 1000,
    "max_storage_gb": 1,
    "max_active_memories": 2500,
    "rate_limit_per_minute": 10
}
```

**Namespace Model**:
```python
# Before
rate_limits: {
    "max_memories_per_month": None,
    "max_api_calls_per_day": None,
    "rate_limit_per_minute": None
}

# After
rate_limits: {
    "max_memory_operations_per_month": None,
    "max_storage_gb": None,
    "max_active_memories": None,
    "rate_limit_per_minute": None
}
```

### 3. ✅ Updated `docs/MULTI_TENANT_SCHEMA_DESIGN.md`

**Before** (generic industry standards):
```markdown
| Tier | Memories/Month | API Calls/Day |
|------|----------------|---------------|
| Trial | 100 | 100 |
| Starter | 1,000 | 1,000 |
| Growth | 10,000 | 10,000 |
```

**After** (actual Papr pricing):
```markdown
| Tier | Memory Operations/Month | Storage | Active Memories | Rate Limit/Min | Price |
|------|------------------------|---------|-----------------|----------------|-------|
| Developer | 1,000 | 1GB | 2,500 | 10 | Free |
| Starter | 50,000 | 10GB | 100,000 | 30 | $100/mo |
| Growth | 750,000 | 100GB | 1,000,000 | 100 | $500/mo |
| Enterprise | Unlimited | Unlimited | Unlimited | 500 | Custom |
```

### 4. ✅ Created `docs/RATE_LIMITS_REFERENCE.md`

New comprehensive reference document with:
- ✅ Complete pricing tier breakdown
- ✅ Field definitions (what counts as a "memory operation")
- ✅ Configuration examples
- ✅ Enforcement points
- ✅ Upgrade paths with ROI
- ✅ Monitoring SQL queries
- ✅ FAQ section

## Pricing Alignment

Now matches exactly what's shown on https://papr.ai/pricing:

| Feature | Developer | Starter | Growth | Enterprise |
|---------|-----------|---------|--------|------------|
| **Memory Operations** | 1K | 50K | 750K | Unlimited |
| **Storage** | 1GB | 10GB | 100GB | Unlimited |
| **Active Memories** | 2,500 | 100,000 | 1,000,000 | Unlimited |
| **Rate Limit/Min** | 10 | 30 | 100 | 500 |
| **Price** | **Free** | **$100/mo** | **$500/mo** | **Custom** |
| **End Users** | Unlimited | Unlimited | Unlimited | Unlimited |
| **Vector + Graph** | ✅ | ✅ | ✅ | ✅ |
| **Support** | Community | Community | Private Slack | Dedicated + SLA |

## Field Name Changes

| Old Field Name | New Field Name | Reason |
|----------------|----------------|--------|
| `max_memories_per_month` | `max_memory_operations_per_month` | More accurate - includes all CRUD operations |
| `max_api_calls_per_day` | (removed) | Redundant with memory operations |
| - | `max_storage_gb` | NEW - from pricing page |
| - | `max_active_memories` | NEW - from pricing page |

## Migration Path

If you have existing organizations in the database with old field names:

```python
# Migration script needed (optional):
# 1. Rename fields in existing Organization documents
db.Organization.updateMany(
  {},
  {
    $rename: {
      "rate_limits.max_memories_per_month": "rate_limits.max_memory_operations_per_month",
      "rate_limits.max_api_calls_per_day": "rate_limits.max_api_calls_per_month"
    },
    $set: {
      "rate_limits.max_storage_gb": 1,  // Default to developer tier
      "rate_limits.max_active_memories": 2500
    }
  }
)

# 2. Update plan_tier for existing orgs
db.Organization.updateMany(
  { plan_tier: "trial" },
  { $set: { plan_tier: "developer" } }
)
```

## Implementation Checklist

### Backend
- [x] Update `config/cloud.yaml` with new limits
- [x] Update `models/parse_server.py` Pydantic models
- [ ] Implement enforcement in `services/rate_limiter.py`
- [ ] Add usage tracking in `services/usage_tracker.py`
- [ ] Create migration script for existing orgs (optional)

### Frontend Dashboard
- [ ] Display current usage vs limits
- [ ] Show upgrade prompts at 80% and 95% usage
- [ ] Add usage charts (memory ops, storage, active memories)
- [ ] Implement upgrade flow to Starter/Growth

### Monitoring
- [ ] Add Amplitude/PostHog events for limit exceeded
- [ ] Create alerts for high usage (80%+)
- [ ] Dashboard for usage trends

## Testing

Test rate limit enforcement:

```bash
# 1. Create test organization with Developer tier
curl -X POST /api/v1/organizations \
  -d '{"name": "Test Org", "plan_tier": "developer"}'

# 2. Try to exceed memory operation limit (1000/month)
# Should get 429 after 1000 operations

# 3. Try to exceed active memories (2500)
# Should get 429 when creating 2501st memory

# 4. Try to exceed storage (1GB)
# Should get 429 when total content exceeds 1GB

# 5. Try to exceed rate limit (10/min)
# Should get 429 after 10 requests in 1 minute
```

## Documentation Links

- **Configuration**: `config/cloud.yaml`
- **Models**: `models/parse_server.py` 
- **Schema Design**: `docs/MULTI_TENANT_SCHEMA_DESIGN.md`
- **Rate Limits Reference**: `docs/RATE_LIMITS_REFERENCE.md`
- **Pricing Page**: https://papr.ai/pricing

## Questions?

Contact: Shawkat Kabbara (shawkat@papr.ai)





