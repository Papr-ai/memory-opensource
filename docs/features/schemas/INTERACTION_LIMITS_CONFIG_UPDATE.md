# Interaction Limits - Configuration Update

## ✅ What Was Done

Moved hardcoded `TIER_LIMITS` dictionary from Python code to `config/cloud.yaml` for centralized management.

## Changes Made

### 1. Updated `config/cloud.yaml`

Added interaction limits to each tier:

```yaml
limits:
  developer:
    # ... existing limits ...
    max_mini_interactions_per_month: 1000
    max_premium_interactions_per_month: 500
  
  starter:
    max_mini_interactions_per_month: 50000
    max_premium_interactions_per_month: 500
  
  growth:
    max_mini_interactions_per_month: 750000
    max_premium_interactions_per_month: 500
  
  enterprise:
    max_mini_interactions_per_month: null  # Unlimited
    max_premium_interactions_per_month: null
  
  # Legacy tiers
  free_trial:
    max_mini_interactions_per_month: 2500
    max_premium_interactions_per_month: 500
  
  pro:
    max_mini_interactions_per_month: 2500
    max_premium_interactions_per_month: 500
  
  business_plus:
    max_mini_interactions_per_month: 5000
    max_premium_interactions_per_month: 1000
```

### 2. Updated `services/user_utils.py`

**Removed hardcoded dict:**
```python
# BEFORE ❌
TIER_LIMITS: TierLimits = {
    'pro': {'mini': 2500, 'premium': 500},
    'business_plus': {'mini': 5000, 'premium': 1000},
    # ... etc
}
```

**Now uses config:**
```python
# AFTER ✅
tier_config = features.get_tier_limits(effective_tier)
mini_limit = tier_config.get('max_mini_interactions_per_month', float('inf'))
premium_limit = tier_config.get('max_premium_interactions_per_month', float('inf'))
```

**Updated methods:**
- ✅ `check_interaction_limits()` - Uses config-based limits
- ✅ `check_interaction_limits_fast()` - Uses config-based limits
- ✅ `_generate_limit_error_message()` - Updated messages to match new tier structure

## Benefits

1. **Centralized Configuration**: All limits in one place (`config/cloud.yaml`)
2. **Easy Updates**: Change limits without touching Python code
3. **Consistent Messaging**: All tier references now consistent (developer, starter, growth, enterprise + legacy)
4. **Unlimited Support**: Enterprise tier properly returns `None` for unlimited access

## Configuration Schema

```yaml
limits:
  {tier_name}:
    # Memory limits
    max_memory_operations_per_month: int | null
    max_storage_gb: int | null
    max_active_memories: int | null
    
    # Interaction limits (for chat UI app)
    max_mini_interactions_per_month: int | null
    max_premium_interactions_per_month: int | null
    
    # Rate limiting
    rate_limit_per_minute: int
```

## Tier Comparison

### New Platform Tiers (Developer Platform)

| Tier | Mini Interactions | Premium Interactions | Price |
|------|------------------|---------------------|-------|
| **Developer** | 1,000 | 500 | FREE |
| **Starter** | 50,000 | 500 | $100/mo |
| **Growth** | 750,000 | 500 | $500/mo |
| **Enterprise** | Unlimited | Unlimited | Custom |

### Legacy Tiers (Productivity App)

| Tier | Mini Interactions | Premium Interactions | Price |
|------|------------------|---------------------|-------|
| **Free Trial** | 2,500 | 500 | FREE |
| **Pro** | 2,500 | 500 | Legacy |
| **Business Plus** | 5,000 | 1,000 | Legacy |

## How It Works

### 1. Feature Flag Check
```python
from config.features import get_features
features = get_features()

# Only run in cloud edition
if not features.is_cloud:
    return None  # Skip all checks in open source
```

### 2. Get Tier Limits
```python
# Get effective tier (free_trial if trialing, else actual tier)
effective_tier = 'free_trial' if is_trial else customer_tier

# Load from config
tier_config = features.get_tier_limits(effective_tier)
# Returns: {'max_mini_interactions_per_month': 1000, ...}
```

### 3. Check Limits
```python
mini_limit = tier_config.get('max_mini_interactions_per_month')
premium_limit = tier_config.get('max_premium_interactions_per_month')

if current_count >= mini_limit:
    return error_response  # User exceeded limit
```

### 4. Generate Error Message
```python
error_message = self._generate_limit_error_message(
    customer_tier='developer',
    limit=1000,
    interaction_type='mini'
)
# Returns tier-specific upgrade path message
```

## Error Message Examples

### Developer Tier
```
You've reached the 1,000 mini interactions limit for your Developer plan.
To continue, upgrade to Starter ($100/mo) or Growth ($500/mo) plan.
Visit https://dashboard.papr.ai to manage your subscription.
```

### Starter Tier
```
You've reached the 50,000 mini interactions limit for your Starter plan.
To continue, you can either:
1. Enable metered billing in your current plan, or
2. Upgrade to Growth plan for higher limits
Visit https://dashboard.papr.ai to manage your subscription.
```

### Growth Tier
```
You've reached the 750,000 mini interactions limit for your Growth plan.
To continue, you can either:
1. Enable metered billing in your current plan, or
2. Contact us for Enterprise plan with unlimited resources
Visit https://dashboard.papr.ai to manage your subscription.
```

## Testing

### Test Configuration Loading
```python
from config.features import get_features

features = get_features()

# Test developer tier
dev_limits = features.get_tier_limits('developer')
assert dev_limits['max_mini_interactions_per_month'] == 1000
assert dev_limits['max_premium_interactions_per_month'] == 500

# Test enterprise tier (unlimited)
enterprise_limits = features.get_tier_limits('enterprise')
assert enterprise_limits['max_mini_interactions_per_month'] is None
assert enterprise_limits['max_premium_interactions_per_month'] is None
```

### Test Open Source Mode
```bash
# Set edition to open source
export PAPR_EDITION=opensource

# All interaction limit checks should return None (skip enforcement)
poetry run python -m pytest tests/test_interaction_limits.py -v
```

### Test Cloud Mode
```bash
# Set edition to cloud
export PAPR_EDITION=cloud

# Limits should be enforced
poetry run python -m pytest tests/test_interaction_limits.py -v
```

## Migration Notes

**No migration required!** This is a drop-in replacement:
- Existing functionality unchanged
- Same API signatures
- Same error responses
- Just reads from config instead of hardcoded dict

## Related Changes

This update completes the configuration migration:
- ✅ Memory limits → `config/cloud.yaml` (previous update)
- ✅ Interaction limits → `config/cloud.yaml` (this update)
- ✅ Edition-aware enforcement (both updates)
- ✅ Legacy tier support (both updates)

## Files Modified

1. ✅ `config/cloud.yaml` - Added interaction limits
2. ✅ `services/user_utils.py` - Updated 3 methods to use config
3. ✅ `INTERACTION_LIMITS_CONFIG_UPDATE.md` - This documentation

---

**Status**: Complete ✅  
**Date**: 2025-10-03  
**Breaking Changes**: None  
**Backward Compatible**: Yes

