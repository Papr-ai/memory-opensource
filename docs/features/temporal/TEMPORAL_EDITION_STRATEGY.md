# Temporal: Cloud vs OSS Strategy

## The Decision: Where Should Temporal Be Available?

You have three strategic options:

---

## Option 1: Cloud-Only Temporal ⭐ RECOMMENDED

### Configuration

```yaml
# config/opensource.yaml
features:
  has_temporal: false
  max_batch_size: 50        # Hard limit for OSS
  max_concurrent_tasks: 10  # Resource protection

# config/cloud.yaml
features:
  has_temporal: true
  max_batch_size: 10000     # No limit with Temporal
  max_concurrent_tasks: 100 # Scale with workers
```

### Pros ✅

1. **Clear Value Proposition**
   ```
   OSS Edition:
   - Process up to 50 memories per batch
   - Great for small teams, personal use
   - Simple deployment (no Temporal needed)
   
   Cloud Edition:
   - Process 10,000+ memories per batch
   - Guaranteed completion with retries
   - Webhook notifications when done
   - No infrastructure management
   ```

2. **Reduced OSS Complexity**
   - No Temporal server to manage
   - No PostgreSQL for Temporal state
   - Simpler docker-compose
   - Lower barrier to entry

3. **Revenue Driver**
   - Enterprises need bulk processing → Must use cloud
   - Clear upgrade path from OSS to cloud
   - Justifies pricing difference

4. **Support Burden**
   - Only support Temporal in controlled cloud environment
   - Fewer variables when debugging issues
   - Easier to monitor and optimize

### Cons ❌

1. **Community Perception**
   - Might be seen as "crippling" OSS version
   - Large self-hosters can't use it

2. **Code Divergence**
   - Need to maintain two code paths
   - More feature flags and conditionals

### Implementation

```python
# routes/memory_routes.py
from config import get_features

async def common_add_memory_batch_handler(...):
    features = get_features()
    batch_size = len(memories)
    
    # Check if batch exceeds OSS limits
    if not features.has_temporal:
        max_batch = features.config.get('max_batch_size', 50)
        if batch_size > max_batch:
            return BatchMemoryResponse.failure(
                error=f"Batch size exceeds limit. OSS supports up to {max_batch} memories. "
                      f"For larger batches, use Papr Cloud: https://papr.ai/cloud",
                code=413
            )
    
    # Cloud edition can use Temporal for large batches
    if features.has_temporal and batch_size > 100:
        return await process_batch_with_temporal(...)
    else:
        return await process_batch_with_background_tasks(...)
```

---

## Option 2: Optional in Both Editions

### Configuration

```yaml
# config/opensource.yaml
features:
  has_temporal: false  # Default off, but can be enabled
  temporal_optional: true
  max_batch_size: 1000  # Higher limit if they configure Temporal

# config/cloud.yaml
features:
  has_temporal: true   # Always on in cloud
  temporal_optional: false
  max_batch_size: 10000
```

### Pros ✅

1. **Maximum Flexibility**
   - OSS users can enable if needed
   - Large enterprises self-hosting can use it
   - Community can contribute Temporal improvements

2. **Unified Codebase**
   - Same code path for both editions
   - Easier to maintain
   - Less feature flag complexity

3. **Community Goodwill**
   - "We're not hiding anything"
   - Transparent about architecture
   - Trust building

### Cons ❌

1. **Support Complexity**
   - Must support Temporal in various configurations
   - Users might misconfigure and blame you
   - More documentation needed

2. **Weaker Differentiation**
   - Less clear reason to use cloud version
   - "Why pay when I can self-host?"

3. **Infrastructure Burden**
   - OSS users must run Temporal + PostgreSQL
   - Deployment complexity increases

### Implementation

```python
# Detection logic
async def common_add_memory_batch_handler(...):
    features = get_features()
    batch_size = len(memories)
    
    # Check if Temporal is available (cloud or self-configured OSS)
    temporal_available = features.has_temporal or await check_temporal_connection()
    
    if temporal_available and batch_size > 100:
        return await process_batch_with_temporal(...)
    else:
        # Enforce limits for non-Temporal setups
        if batch_size > 50:
            return BatchMemoryResponse.failure(
                error="Large batches require Temporal. Configure Temporal or use Papr Cloud.",
                code=413
            )
        return await process_batch_with_background_tasks(...)
```

### Documentation (README)

```markdown
## Large Batch Processing (Optional)

For processing > 50 memories per batch, Temporal is required.

### Cloud Edition
Temporal is pre-configured and managed. Just use it!

### OSS Edition
To enable large batches, configure Temporal:

1. Run Temporal server:
   ```bash
   docker-compose -f docker-compose-temporal.yaml up -d
   ```

2. Enable in config:
   ```yaml
   # config/opensource.yaml
   features:
     has_temporal: true
   ```

3. Set environment:
   ```bash
   TEMPORAL_HOST=localhost:7233
   ```

See `docs/TEMPORAL_SETUP.md` for details.
```

---

## Option 3: OSS-Only Alternative (Inngest, BullMQ)

### Strategy

Provide a **simpler alternative** for OSS, keep Temporal for cloud:

```yaml
# config/opensource.yaml
features:
  has_temporal: false
  has_bullmq: true  # Simpler queue system
  max_batch_size: 500

# config/cloud.yaml
features:
  has_temporal: true
  has_bullmq: false
  max_batch_size: 10000
```

### Pros ✅

1. **Best of Both Worlds**
   - OSS gets queue system without Temporal complexity
   - Cloud gets enterprise-grade Temporal
   - Clear differentiation

2. **Lower OSS Barrier**
   - BullMQ only needs Redis (already required)
   - No additional infrastructure
   - Simpler to configure

### Cons ❌

1. **Two Systems to Maintain**
   - BullMQ code path for OSS
   - Temporal code path for cloud
   - More complexity internally

2. **Different Guarantees**
   - BullMQ: Durable but not as robust as Temporal
   - Temporal: Enterprise-grade guarantees
   - Different failure modes

---

## Recommendation: **Option 1 (Cloud-Only)** ⭐

### Why Cloud-Only Is Best

```
┌─────────────────────────────────────────────────────────────┐
│                    Use Case Mapping                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Small Teams / Personal Use                                 │
│  ├── Process: < 50 memories at a time                       │
│  ├── Deployment: Single server                              │
│  └── Solution: OSS (no Temporal needed) ✓                   │
│                                                              │
│  Medium Companies                                            │
│  ├── Process: 50-500 memories at a time                     │
│  ├── Deployment: Small cluster                              │
│  └── Solution: Cloud (managed Temporal) ✓                   │
│                                                              │
│  Enterprises                                                 │
│  ├── Process: 1,000-10,000+ memories at a time             │
│  ├── Deployment: Multi-region                               │
│  └── Solution: Cloud (full Temporal power) ✓               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Plan

#### Phase 1: Set Limits (Now)

```python
# config/opensource.yaml
features:
  has_temporal: false
  max_batch_size: 50
  max_concurrent_processing: 10
  
messaging:
  batch_limit_message: |
    For processing > 50 memories, use Papr Cloud with guaranteed 
    delivery and webhook notifications: https://papr.ai/cloud

# config/cloud.yaml  
features:
  has_temporal: true
  max_batch_size: 10000  # Effectively unlimited
  max_concurrent_processing: 1000
```

#### Phase 2: Enforce Limits

```python
# routes/memory_routes.py
async def common_add_memory_batch_handler(...):
    from config import get_features
    
    features = get_features()
    batch_size = len(memories)
    
    # Check batch size limits
    max_batch = features.config.get('max_batch_size', 50)
    if batch_size > max_batch:
        # Provide helpful upgrade message
        upgrade_message = features.config.get('messaging', {}).get('batch_limit_message', '')
        
        return BatchMemoryResponse.failure(
            error=f"Batch size ({batch_size}) exceeds limit ({max_batch})",
            code=413,
            details={
                "max_batch_size": max_batch,
                "upgrade_info": upgrade_message,
                "upgrade_url": "https://papr.ai/cloud"
            }
        )
    
    # Use appropriate processing based on edition
    if features.has_temporal and batch_size > 100:
        logger.info(f"Using Temporal for batch of {batch_size} (cloud edition)")
        return await process_batch_with_temporal(...)
    else:
        logger.info(f"Using background tasks for batch of {batch_size}")
        return await process_batch_with_background_tasks(...)
```

#### Phase 3: Document Clearly

```markdown
# README.md

## Batch Processing Limits

### Open Source Edition
- **Batch size**: Up to 50 memories per request
- **Processing**: In-memory background tasks
- **Use case**: Personal use, small teams, development

### Cloud Edition
- **Batch size**: Up to 10,000 memories per request
- **Processing**: Durable workflows with Temporal
- **Features**:
  - ✓ Guaranteed completion (survives server restarts)
  - ✓ Automatic retries on failures
  - ✓ Webhook notifications when done
  - ✓ Progress tracking
  - ✓ No infrastructure management

**Need bulk processing?** [Try Papr Cloud](https://papr.ai/cloud)
```

---

## Pricing Justification

With Temporal as cloud-only:

```
Open Source (Free)
├── Single memories: Unlimited
├── Batch processing: Up to 50
├── Infrastructure: You manage
└── Support: Community

Cloud Starter ($99/mo)
├── Single memories: Unlimited
├── Batch processing: Up to 1,000 with Temporal
├── Infrastructure: Managed
├── Support: Email
└── Webhooks: Included

Cloud Business ($499/mo)
├── Single memories: Unlimited
├── Batch processing: Up to 10,000 with Temporal
├── Infrastructure: Managed + redundant
├── Support: Priority
├── SLA: 99.9% uptime
└── Advanced: Multi-region, custom workflows
```

**Value Prop**: "Need to process thousands of documents? Temporal guarantees every document is processed, even if servers fail."

---

## Migration Path for OSS → Cloud

Make it easy:

```python
# CLI command for OSS users
$ papr check-limits

Current Usage:
  Average batch size: 75 memories
  Max batch attempted: 120 memories
  Failed batches: 3 (batch too large)

Recommendation:
  Your usage exceeds OSS limits. Consider Papr Cloud:
  
  Benefits:
  ✓ Process batches up to 10,000 memories
  ✓ Guaranteed delivery with Temporal
  ✓ Webhook notifications
  ✓ Managed infrastructure
  
  Start free trial: https://papr.ai/cloud/trial
```

---

## Alternative: Hybrid Approach

If you want to be **very** generous to OSS:

```yaml
# config/opensource.yaml
features:
  has_temporal: false
  max_batch_size: 50              # Without Temporal
  max_batch_size_with_temporal: 1000  # If they set up Temporal
  
  # Allow them to configure Temporal but don't support it officially
  temporal_experimental: true
  
documentation:
  temporal_setup: |
    Temporal support is experimental in OSS. We recommend Papr Cloud
    for production batch processing, but you can configure Temporal:
    See docs/TEMPORAL_SETUP_OSS.md (community-supported)
```

This lets them do it, but you're not officially supporting it.

---

## Final Recommendation

### **Go with Cloud-Only Temporal (Option 1)**

**Reasoning:**

1. **Clear differentiation**: Enterprises need bulk → must pay
2. **Simpler OSS**: Lower barrier, easier onboarding
3. **Revenue driver**: Justifies cloud pricing
4. **Support focus**: Only support Temporal in cloud
5. **Upgrade path**: Natural progression OSS → Cloud

**Messaging:**

```
"Open source is great for getting started and small-scale use.
 When you need enterprise-scale batch processing with guarantees,
 Papr Cloud provides managed Temporal infrastructure that just works."
```

**Exception:**

If an enterprise wants to self-host AND needs Temporal:
- Offer **Enterprise Self-Hosted** tier ($$$)
- Include Temporal setup & support
- Still profitable because they pay for support

---

## Implementation Checklist

- [ ] Set `has_temporal: false` in `config/opensource.yaml`
- [ ] Set `max_batch_size: 50` in OSS config
- [ ] Add batch size validation in routes
- [ ] Add helpful upgrade messaging
- [ ] Document limits in README
- [ ] Create comparison page on website
- [ ] Add "Upgrade to Cloud" banner in OSS UI for large batches
- [ ] Track batch size metrics to identify upgrade candidates

---

## Summary Table

| Feature | OSS | Cloud Starter | Cloud Business |
|---------|-----|---------------|----------------|
| **Single memory** | ✓ Unlimited | ✓ Unlimited | ✓ Unlimited |
| **Batch (no Temporal)** | ✓ Up to 50 | ✓ Up to 50 | ✓ Up to 50 |
| **Batch (with Temporal)** | ✗ Not available | ✓ Up to 1,000 | ✓ Up to 10,000 |
| **Guaranteed delivery** | ✗ | ✓ | ✓ |
| **Webhook notifications** | Basic | ✓ Reliable | ✓ Reliable |
| **Infrastructure** | Self-manage | Managed | Managed + HA |
| **Support** | Community | Email | Priority |

**This gives you a clear, defensible pricing strategy!**

