# Rate Limits Reference

Quick reference for Papr Memory rate limits based on pricing tiers.

## Pricing Tiers (from https://papr.ai/pricing)

| Tier | Price | Memory Ops/Month | Storage | Active Memories | Rate Limit/Min |
|------|-------|------------------|---------|-----------------|----------------|
| **Developer** | Free | 1,000 | 1GB | 2,500 | 10 |
| **Starter** | $100/mo | 50,000 | 10GB | 100,000 | 30 |
| **Growth** | $500/mo | 750,000 | 100GB | 1,000,000 | 100 |
| **Enterprise** | Custom | Unlimited | Unlimited | Unlimited | 500 |

## Field Definitions

### Memory Operations
**Field**: `max_memory_operations_per_month`

Counts all API operations that interact with memories:
- `POST /memories` - Create new memory
- `GET /memories` - Retrieve memories  
- `GET /memories/:id` - Get single memory
- `PUT /memories/:id` - Update memory
- `DELETE /memories/:id` - Delete memory
- `POST /memories/search` - Search memories

**Not counted**:
- Authentication requests
- Health checks
- Metadata-only queries

### Storage
**Field**: `max_storage_gb`

Total storage capacity for:
- Memory content (text, embeddings)
- Metadata (title, tags, topics)
- Context data
- Custom metadata
- File attachments (if applicable)

### Active Memories
**Field**: `max_active_memories`

Maximum number of non-deleted memories that can exist in the organization.

**Counts**:
- All memories with `deleted: false` or no deleted flag
- Across all namespaces in the organization

**Does not count**:
- Soft-deleted memories
- Archived memories (if implemented)

### Rate Limit Per Minute
**Field**: `rate_limit_per_minute`

Burst protection - maximum API requests per minute from a single API key.

Uses sliding window algorithm.

## Configuration Files

### `config/cloud.yaml`

```yaml
limits:
  developer:
    max_memory_operations_per_month: 1000
    max_storage_gb: 1
    max_active_memories: 2500
    rate_limit_per_minute: 10
  
  starter:
    max_memory_operations_per_month: 50000
    max_storage_gb: 10
    max_active_memories: 100000
    rate_limit_per_minute: 30
  
  growth:
    max_memory_operations_per_month: 750000
    max_storage_gb: 100
    max_active_memories: 1000000
    rate_limit_per_minute: 100
  
  enterprise:
    max_memory_operations_per_month: null  # Unlimited
    max_storage_gb: null
    max_active_memories: null
    rate_limit_per_minute: 500
```

### Organization Rate Limits (MongoDB/Parse)

```javascript
// Organization document
{
  "plan_tier": "growth",
  "rate_limits": {
    "max_memory_operations_per_month": 750000,
    "max_storage_gb": 100,
    "max_active_memories": 1000000,
    "rate_limit_per_minute": 100
  }
}
```

### Namespace Rate Limits (Overrides)

```javascript
// Production namespace - higher rate limit
{
  "name": "acme-production",
  "environment_type": "production",
  "rate_limits": {
    "max_memory_operations_per_month": null,  // Inherit 750K
    "max_storage_gb": null,  // Inherit 100GB
    "max_active_memories": null,  // Inherit 1M
    "rate_limit_per_minute": 200  // Override: 2x for production
  }
}

// Development namespace - lower limits
{
  "name": "acme-development",
  "environment_type": "development",
  "rate_limits": {
    "max_memory_operations_per_month": 10000,  // Override: 1% for testing
    "max_storage_gb": 1,  // Override: 1GB for dev
    "max_active_memories": 10000,  // Override: 1% for testing
    "rate_limit_per_minute": null  // Inherit 100 from org
  }
}
```

## Enforcement Points

### 1. API Gateway (Rate Limiting)
```python
# middleware/rate_limiter.py
if requests_this_minute > organization.rate_limits.rate_limit_per_minute:
    raise HTTPException(429, "Rate limit exceeded")
```

### 2. Memory Creation
```python
# services/memory_service.py
if org_memory_ops_this_month >= organization.rate_limits.max_memory_operations_per_month:
    raise HTTPException(429, "Monthly operation limit exceeded")

if total_active_memories >= organization.rate_limits.max_active_memories:
    raise HTTPException(429, "Active memory limit exceeded")
```

### 3. Storage Check
```python
# services/storage_service.py
if org_storage_used_gb >= organization.rate_limits.max_storage_gb:
    raise HTTPException(429, "Storage limit exceeded")
```

## Upgrade Paths

### Developer → Starter
- 50x more memory operations (1K → 50K)
- 10x more storage (1GB → 10GB)
- 40x more active memories (2,500 → 100,000)
- 3x higher rate limit (10 → 30 req/min)
- **Cost**: $100/month

### Starter → Growth  
- 15x more memory operations (50K → 750K)
- 10x more storage (10GB → 100GB)
- 10x more active memories (100K → 1M)
- 3.3x higher rate limit (30 → 100 req/min)
- **Cost**: $500/month (+$400)

### Growth → Enterprise
- Unlimited everything
- 5x higher rate limit (100 → 500 req/min)
- Priority support
- Custom SLA
- On-prem deployment options
- **Cost**: Custom pricing

## Monitoring

### Track Usage
```sql
-- Memory operations this month
SELECT COUNT(*) FROM MemoryOperationLog
WHERE organization_id = 'org_xxx'
  AND created_at >= date_trunc('month', CURRENT_DATE);

-- Active memories
SELECT COUNT(*) FROM Memory
WHERE organization_id = 'org_xxx'
  AND (deleted IS NULL OR deleted = false);

-- Storage used
SELECT SUM(content_size + metadata_size) / (1024*1024*1024) as gb_used
FROM Memory
WHERE organization_id = 'org_xxx';
```

### Alert When Approaching Limits
```python
# Send email/webhook when:
usage_percent = (current_usage / limit) * 100

if usage_percent >= 80:
    send_alert(f"You've used {usage_percent}% of your {limit_type} limit")
    
if usage_percent >= 95:
    send_urgent_alert(f"You're approaching your {limit_type} limit")
```

## FAQ

**Q: What happens if I exceed my limit?**
A: API requests will return `429 Too Many Requests` with details about which limit was exceeded.

**Q: Can I temporarily exceed limits?**
A: No. Limits are hard caps. Upgrade to a higher tier for more capacity.

**Q: Do deleted memories count toward my active memory limit?**
A: No. Only non-deleted memories count.

**Q: Can I have different limits for different namespaces?**
A: Yes! Namespace rate_limits can override organization defaults.

**Q: How do I see my current usage?**
A: Check the developer dashboard at `/dashboard/usage` (coming soon).

**Q: What counts as a "memory operation"?**
A: Create, read, update, delete, and search operations. Authentication and health checks don't count.

## Related Documents

- [Multi-Tenant Schema Design](./MULTI_TENANT_SCHEMA_DESIGN.md) - Full schema with Parse pointers
- [Pricing Page](https://papr.ai/pricing) - Public pricing tiers
- `config/cloud.yaml` - Rate limit configuration
- `models/parse_server.py` - Pydantic models with rate_limits fields

