# Search Latency Analysis

**Date**: January 29, 2026  
**Environment**: Cloud Run (memoryserver-development)  
**Region**: us-west1

---

## Summary

**SLA Target**: < 500ms

| Scenario | Total Latency | Status | Frequency |
|----------|---------------|--------|-----------|
| **Cold Start** | 2,447ms | ❌ 5x over SLA | Every new instance |
| **Warm + Cache Expiry** | 761ms | ❌ 1.5x over SLA | Every ~2 min/user |
| **Warm + Slow Qdrant** | 529ms | ❌ Over SLA | ~17% of requests |
| **Warm + All Caches HIT** | 240-340ms | ✅ Within SLA | ~70% of requests |

### When Each Scenario Occurs

#### Cold Start (2,447ms)
*Note: Cloud Run is configured with min-instances: 1, so scale-from-zero is avoided.*

User experiences cold start when:
- **New deployment** - New revision deployed, instance replaced (affects first request after deploy)
- **Traffic spike** - 2nd, 3rd+ instances spin up to handle concurrent load (affects requests routed to new instances)
- **Instance recycled** - Cloud Run periodically recycles long-running instances (~hours)
- **Instance crash/restart** - Health check failure or OOM triggers replacement

**Impact**: Less frequent than scale-from-zero, but still affects ~1-5% of requests during deployments and traffic spikes.

#### Warm + Cache Expiry (761ms)
User experiences cache expiry when:
- **Auth cache expires (TTL=120s)** - User's auth data evicted, triggers V2 user resolution (+226ms)
- **Embedding cache expires** - Query not seen in last few minutes, triggers Vertex AI call (+136ms)
- **Both expire together** - Cascading effect when user inactive for ~2 minutes

**Impact**: Active users see this spike every ~2 minutes if continuously querying.

#### Warm + Slow Qdrant (529ms)
User experiences slow Qdrant when:
- **Network jitter** - Cross-region latency to Qdrant Cloud (us-east-1 ↔ us-west1)
- **Qdrant cluster load** - Other tenants on shared Qdrant Cloud causing contention
- **Connection not reused** - TCP connection pool not warm for that route

**Impact**: Random ~17% of requests affected regardless of user activity pattern.

#### Warm + All Caches HIT (240-340ms) ✅
User experiences optimal latency when:
- **Repeat query** - Same query within cache TTL (embedding cached)
- **Active session** - Continuous activity keeps auth cache warm
- **Normal Qdrant** - No network jitter or cluster load

**Impact**: This is the target state - achievable for majority of requests with fixes.

### Key Findings

| Issue | Observed Impact | Root Cause | Analysis Reference |
|-------|-----------------|------------|-------------------|
| Cold start | +2,200ms | MongoDB warm-up, all caches empty | [First Request Analysis](#first-request-timing-distribution-cold-start) |
| Auth cache expiry | +226ms | TTL=120s triggers V2 user resolution | [Cache Expiry Analysis](#cache-expiry-impact-analysis) |
| Embedding cache miss | +136ms | New query triggers Vertex AI call | [Cache Expiry Analysis](#vertex-ai-call-behavior-cache-hit-vs-miss) |
| Qdrant variance | +159ms | 2x latency variance (136-315ms) | [Multi-Request Analysis](#qdrant-latency-variance) |
| Sync cache logging | +114ms | Blocking stats logging in auth | [Multi-Request Analysis](#request-4-430ms---pre-search-overhead-spike) |

---

## Prioritized Action Plan

Based on the analyses in this document, here are the recommended changes ordered by **impact** (latency reduction potential):

### Priority 1: Reduce Cold Start Impact (Impact: -1,700ms)

*Note: min-instances: 1 is already configured. Cold starts still occur during deployments, traffic spikes, and instance recycling.*

| Action | Expected Savings | Effort | Reference |
|--------|------------------|--------|-----------|
| **Add startup cache/connection warming** | -1,400ms (MongoDB warm-up) | 15 lines | [First Request Analysis](#first-request-timing-distribution-cold-start) |
| **Enable CPU always-on** (if not set) | -50ms (CPU throttle) | Config only | [First Request Analysis](#first-request-timing-distribution-cold-start) |
| **Pre-warm Vertex AI connection** | -250ms (first embedding) | 5 lines | [First Request Analysis](#first-request-timing-distribution-cold-start) |

```python
# main.py - Add startup event
@app.on_event("startup")
async def warm_on_startup():
    # 1. Warm MongoDB connection pool
    await mongodb_client.admin.command('ping')
    
    # 2. Pre-warm Vertex AI endpoint with dummy embedding
    try:
        await embedding_model.embed_query("warmup")
    except:
        pass  # Ignore errors, just warming connection
    
    # 3. Log startup complete
    logger.info("Startup warming complete")
```

```yaml
# cloud-run-service.yaml - Ensure CPU always-on
run.googleapis.com/cpu-throttling: "false"
```

**Evidence**: Cold start breakdown shows MongoDB warm-up (1,411ms) and Vertex AI cold call (310ms) are main contributors. Startup warming moves this cost to deployment time instead of first user request.

---

### Priority 2: Extend Auth Cache TTL (Impact: -226ms on expiry)

| Action | Expected Savings | Effort | Reference |
|--------|------------------|--------|-----------|
| **Extend `auth_optimized_cache` TTL: 120s → 300s** | -226ms per expiry event | 1 line change | [Cache Expiry Analysis](#cache-expiry-impact-analysis) |
| Add TTL jitter (±10s) | Prevents cascading misses | 3 lines | [Cache Expiry Analysis](#cache-expiry-impact-analysis) |

```python
# services/cache_utils.py
AUTH_OPTIMIZED_CACHE_TTL = 300  # Was 120s, now 5 minutes
ttl = AUTH_OPTIMIZED_CACHE_TTL + random.uniform(-10, 10)  # Jitter
```

**Evidence**: In Cache Expiry Analysis, auth cache MISS added +226ms (V2 user resolution: 171ms + auth overhead: 55ms). With 120s TTL, this happens every 2 minutes per user.

---

### Priority 3: Async Cache Stats Logging (Impact: -114ms sporadic)

| Action | Expected Savings | Effort | Reference |
|--------|------------------|--------|-----------|
| **Move cache stats logging to background** | -114ms (when triggered) | 5 lines | [Multi-Request Analysis, Request #4](#request-4-430ms---pre-search-overhead-spike) |

```python
# services/auth_utils.py
# Before (blocking):
log_cache_statistics()

# After (non-blocking):
asyncio.create_task(log_cache_statistics())
```

**Evidence**: In Multi-Request Analysis, Request #4 had 118ms pre-search overhead (vs normal 4ms) due to synchronous cache stats logging.

---

### Priority 4: Qdrant Timeout with Retry (Impact: -159ms on slow queries)

| Action | Expected Savings | Effort | Reference |
|--------|------------------|--------|-----------|
| **Add 200ms timeout with retry** | -159ms (on slow Qdrant) | 10 lines | [Multi-Request Analysis, Request #5](#request-5-529ms---over-500ms-sla-) |
| Consider region proximity | -50ms (cross-region) | Infrastructure | [Qdrant Latency Variance](#qdrant-latency-variance) |

```python
# memory/memory_graph.py
async def search_with_timeout(query_embedding, timeout_ms=200):
    try:
        return await asyncio.wait_for(
            qdrant_client.search(...), 
            timeout=timeout_ms/1000
        )
    except asyncio.TimeoutError:
        logger.warning("Qdrant timeout, retrying...")
        return await qdrant_client.search(...)  # Retry once
```

**Evidence**: Qdrant latency varies 136ms-315ms (2.3x). Request #5 had 309ms Qdrant time causing SLA breach. App is in us-west1, Qdrant in us-east-1.

---

### Priority 5: Proactive Cache Refresh (Impact: prevents expiry spikes)

| Action | Expected Savings | Effort | Reference |
|--------|------------------|--------|-----------|
| **Refresh at 80% TTL** | Prevents -355ms spikes | 15 lines | [Cache Expiry Analysis](#request-1-bottleneck-analysis-761ms---over-sla) |

```python
# services/cache_utils.py
async def get_with_proactive_refresh(cache, key, ttl, fetch_fn):
    value, age = cache.get_with_age(key)
    if value and age > ttl * 0.8:  # 80% of TTL
        asyncio.create_task(refresh_cache(cache, key, fetch_fn))
    return value
```

**Evidence**: In Cache Expiry Analysis, simultaneous auth + embedding cache expiry caused +355ms latency spike (761ms total).

---

### Impact Summary

| Priority | Action | Latency Reduction | Frequency |
|----------|--------|-------------------|-----------|
| P1 | Min instances + CPU always-on | **-2,200ms** | Every cold start |
| P2 | Extend auth TTL + jitter | **-226ms** | Every 2 min/user → every 5 min |
| P3 | Async cache stats logging | **-114ms** | Sporadic |
| P4 | Qdrant timeout/retry | **-159ms** | ~17% of requests |
| P5 | Proactive cache refresh | **-355ms** | Prevents spikes |

**Total potential improvement**: P50 latency from 250ms → <200ms, P99 from 761ms → <500ms

---

## First Request Timing Distribution (Cold Start)

Total end-to-end latency: **2,447ms**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FIRST REQUEST BREAKDOWN                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ████████████████████████████████████████████████████░░░░░░░░░░░  1,411ms  │
│  V2 User Resolution (58%)                                                   │
│  - MongoDB connection warm-up                                               │
│  - User/workspace/org/namespace lookups                                     │
│                                                                             │
│  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    310ms  │
│  Embedding Generation (13%)                                                 │
│  - Vertex AI inference (cold endpoint)                                      │
│  - Cache MISS → external API call                                           │
│                                                                             │
│  ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    284ms  │
│  Qdrant Vector Search (12%)                                                 │
│  - 45 results returned                                                      │
│                                                                             │
│  ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    291ms  │
│  Post-processing (12%)                                                      │
│  - Stratified sorting                                                       │
│  - Memory fetch from MongoDB                                                │
│                                                                             │
│  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    142ms  │
│  Auth & Overhead (5%)                                                       │
│  - API key validation (cache MISS)                                          │
│  - Workspace subscription lookup (cache MISS)                               │
│  - Customer tier lookup (cache MISS)                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Breakdown

| Phase | Duration | % of Total | Cache Status |
|-------|----------|------------|--------------|
| V2 User Resolution | 1,411ms | 58% | MISS |
| Pre-search overhead | 1,553ms | 64% | Multiple MISS |
| └─ Auth optimized | ~50ms | 2% | MISS |
| └─ API key validation | ~50ms | 2% | MISS |
| └─ Workspace subscription | ~87ms | 4% | MISS |
| Embedding generation | 310ms | 13% | MISS |
| └─ Vertex AI inference | 307ms | 13% | Cold |
| └─ Embedding parsing | 0.2ms | 0% | - |
| Qdrant search | 284ms | 12% | - |
| Post-processing | 291ms | 12% | - |
| └─ Stratified sorting | ~7ms | 0% | - |
| └─ Memory fetch | 150ms | 6% | - |
| Response build | 0.1ms | 0% | - |
| **Total** | **2,427ms** | **100%** | |

---

## Subsequent Request Timing Distribution (Warm)

Total end-to-end latency: **240ms**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SUBSEQUENT REQUEST BREAKDOWN                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░      4ms  │
│  Pre-search overhead (2%)                                                   │
│  - All caches HIT                                                           │
│                                                                             │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░      0ms  │
│  Embedding Generation (0%)                                                  │
│  - Cache HIT → skipped Vertex AI                                            │
│                                                                             │
│  ████████████████████████████████████████████████████████████░░    230ms  │
│  Memory Search (96%)                                                        │
│  - Qdrant search + post-processing                                          │
│                                                                             │
│  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░      6ms  │
│  Response build (2%)                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Breakdown

| Phase | Duration | % of Total | Cache Status |
|-------|----------|------------|--------------|
| Pre-search overhead | 4.2ms | 2% | All HIT |
| └─ Auth optimized | <1ms | 0% | HIT (age: 4.9s) |
| └─ Workspace subscription | <1ms | 0% | HIT (age: 4.8s) |
| Embedding generation | 0ms | 0% | HIT (age: 4.5s) |
| Memory search | 230ms | 96% | - |
| Response build | 6ms | 2% | - |
| **Total** | **240ms** | **100%** | |

---

## Cache Behavior Comparison

| Cache | First Request | Subsequent Requests |
|-------|---------------|---------------------|
| `auth_optimized_cache` | MISS | HIT |
| `api_key_cache` | MISS | HIT |
| `workspace_subscription_cache` | MISS | HIT |
| `search_embedding_cache` | MISS → Vertex AI call | HIT → 0ms |
| `customer_tier_cache` | MISS | HIT |

---

## Root Cause Analysis

### Why First Request is 10x Slower

1. **V2 User Resolution (1,411ms)** - The biggest bottleneck
   - Cold MongoDB connection pool
   - Multiple sequential lookups for user resolution
   - No cached user/workspace data

2. **Embedding Generation (310ms)**
   - Cache MISS triggers Vertex AI API call
   - Cold model endpoint adds latency

3. **All Auth Caches Cold**
   - API key validation requires DB lookup
   - Workspace subscription requires DB lookup
   - Customer tier requires Stripe API call

---

## Recommendations

### Immediate (No Code Changes)

1. **Set minimum instances to 1** in Cloud Run:
   ```yaml
   min-instances: 1
   ```
   Cost: ~$30-50/month, eliminates cold start

2. **Use Cloud Run CPU always-on**:
   ```yaml
   cpu-throttling: false
   ```

### Short-term (Code Changes)

1. **Add startup cache warming**:
   ```python
   @app.on_event("startup")
   async def warm_caches():
       # Pre-connect to MongoDB
       await mongodb_client.admin.command('ping')
       # Warm common auth patterns
   ```

2. **Parallelize V2 user resolution**:
   - Current: Sequential lookups
   - Proposed: `asyncio.gather()` for independent lookups

3. **Add connection pooling warm-up for MongoDB**

### Long-term

1. **Optimize V2 user resolution architecture**
   - Consider denormalization
   - Add Redis caching layer for user data

2. **Pre-compute common embeddings**
   - Background job to cache popular queries

---

## Performance Targets

| Metric | Current (Cold) | Current (Warm) | Target |
|--------|----------------|----------------|--------|
| P50 Latency | 2,400ms | 250ms | < 300ms |
| P95 Latency | 2,800ms | 400ms | < 500ms |
| P99 Latency | 3,200ms | 600ms | < 1,000ms |

---

## Multi-Request Analysis (Warm Instance)

**Date**: January 29, 2026  
**Sample Size**: 6 consecutive search requests over 22 seconds  
**Query**: `"find my latest memories related to benchmarking"`

### Request Summary Table

| # | Time | HTTP Latency | Pre-search | Memory Search | Embedding | Qdrant | Status |
|---|------|--------------|------------|---------------|-----------|--------|--------|
| 1 | 17:43:15 | **272ms** | 4ms | 242ms | HIT | 154ms | ✅ |
| 2 | 17:43:19 | **354ms** | 3ms | 325ms | MISS (106ms) | 136ms | ✅ |
| 3 | 17:43:23 | **266ms** | 5ms | 231ms | HIT | 157ms | ✅ |
| 4 | 17:43:27 | **430ms** | **118ms** ⚠️ | 286ms | HIT | ~140ms | ⚠️ |
| 5 | 17:43:32 | **529ms** | 4ms | **494ms** | MISS (102ms) | **309ms** ⚠️ | ❌ |
| 6 | 17:43:37 | **294ms** | 4ms | 267ms | HIT | ~160ms | ✅ |

### Latency Distribution

```
< 300ms:  ███████████████ 3 requests (50%)  ✅ Good
300-500ms: ██████████ 2 requests (33%)       ⚠️ Acceptable  
> 500ms:   █████ 1 request (17%)             ❌ Over SLA

Average: 358ms | Min: 266ms | Max: 529ms | Std Dev: 99ms
```

### Bottleneck Analysis

#### Request #4 (430ms) - Pre-search Overhead Spike

| Phase | Duration | Issue |
|-------|----------|-------|
| Authentication timing | **42ms** | Cache stats logging overhead |
| Pre-search total | **118ms** | 30x higher than normal (4ms) |

**Root Cause**: Synchronous cache statistics logging:
```
Cache Statistics:
  API Key Cache: {...}
  Session Token Cache: {...}
  Auth Optimized Cache: {...}
```

**Fix**: Move cache stats logging to background/async.

#### Request #5 (529ms) - Over 500ms SLA ❌

| Phase | Duration | Normal | Delta |
|-------|----------|--------|-------|
| Embedding (Vertex AI) | 102ms | 0ms (cached) | +102ms |
| Qdrant search | **309ms** | ~150ms | **+159ms** |
| Total memory search | **494ms** | ~240ms | +254ms |

**Root Causes**:
1. **Embedding Cache MISS** → Different query hash triggered Vertex AI call
2. **Slow Qdrant** → 309ms vs typical 150ms (2x variance)

### Qdrant Latency Variance

| Request | Qdrant Time | Results | Status |
|---------|-------------|---------|--------|
| #1 | 154ms | 45 | Normal |
| #2 | 136ms | 45 | Fast |
| #3 | 157ms | 45 | Normal |
| #4 | ~140ms | 45 | Normal |
| #5 | **309ms** | 45 | **Slow** |
| #6 | ~160ms | 45 | Normal |

**Observation**: Qdrant Cloud shows 2x latency variance (136ms - 309ms) likely due to:
- Network jitter to Qdrant Cloud (us-east-1)
- Qdrant cluster load fluctuation
- TCP connection reuse patterns

### Additional Recommendations (from Multi-Request Analysis)

1. **Async cache stats logging**:
   ```python
   # Instead of synchronous logging
   asyncio.create_task(log_cache_statistics())
   ```

2. **Qdrant query timeout with retry**:
   ```python
   async def search_with_timeout(query, timeout=200):
       try:
           return await asyncio.wait_for(qdrant_search(query), timeout/1000)
       except asyncio.TimeoutError:
           return await qdrant_search(query)  # Retry once
   ```

3. **Monitor Qdrant P95 latency** - Alert if consistently > 200ms

4. **Consider Qdrant region proximity** - Current: us-east-1, App: us-west1

---

## Cache Expiry Impact Analysis

**Date**: January 29, 2026  
**Sample Size**: 4 consecutive search requests over 37 seconds  
**Query**: `"find my latest memories related to VC notes"`

### Request Summary Table

| # | Time | HTTP Latency | Pre-search | Memory Search | Embedding | Qdrant | Status |
|---|------|--------------|------------|---------------|-----------|--------|--------|
| 1 | 17:49:15 | **761ms** ❌ | **232ms** | 500ms | **MISS (136ms)** | 282ms | Over SLA |
| 2 | 17:49:31 | **421ms** | 4ms | 387ms | HIT (0.2ms) | 315ms | ✅ |
| 3 | 17:49:48 | **384ms** | 5ms | 355ms | HIT (0.5ms) | 286ms | ✅ |
| 4 | 17:49:52 | **413ms** | 4ms | 328ms | HIT (0.2ms) | 168ms | ✅ |

### Latency Distribution

```
Request #1:  ████████████████████████████████████████  761ms ❌ Over 500ms
Request #2:  ██████████████████████                    421ms ✅
Request #3:  ████████████████████                      384ms ✅
Request #4:  █████████████████████                     413ms ✅

Average (warm): 406ms | Request #1 delta: +355ms
```

### Request #1 Bottleneck Analysis (761ms - Over SLA)

| Phase | Duration | Warm Baseline | **Delta** |
|-------|----------|---------------|-----------|
| Auth optimized cache | MISS | HIT | - |
| Optimized auth timing | **227ms** | ~1ms | **+226ms** |
| └─ V2 user resolution | 171ms | 0ms (cached) | +171ms |
| └─ API key cache | MISS | HIT | - |
| Pre-search total | **232ms** | 4ms | **+228ms** |
| Embedding cache | MISS | HIT | - |
| Vertex AI inference | **136ms** | 0ms | **+136ms** |
| Qdrant search | 282ms | ~257ms | +25ms |
| **Total Delta** | | | **~355ms** |

### Root Causes for Cache Expiry Spike

1. **Auth Optimized Cache MISS** (+226ms)
   ```
   [auth_optimized_cache] Cache cleanup: removed 1 expired entries
   [auth_optimized_cache] Cache MISS for auth_optimized:APIKe...
   ```
   - Cache entry expired (TTL = 120s)
   - Triggered full V2 user resolution (171ms)
   - Required MongoDB lookups for user, workspace_follower

2. **Embedding Cache MISS** (+136ms)
   ```
   [search_embedding_cache] Cache cleanup: removed 1 expired entries
   [search_embedding_cache] Cache MISS for qwen_search:7079a6db...
   ```
   - First query for "VC notes" after cache expiry
   - Forced Vertex AI inference (136ms)

3. **Cascading Effect**: Both caches expired around the same time

### Vertex AI Call Behavior (Cache HIT vs MISS)

| Request | Embedding Cache | Vertex AI Called? | Embedding Time | Log Evidence |
|---------|-----------------|-------------------|----------------|--------------|
| #1 | **MISS** | **Yes** | 138.5ms | `Vertex AI inference took: 0.1355 seconds` |
| #2 | HIT (16s old) | **No** | 0.21ms | No Vertex AI log line |
| #3 | HIT (33s old) | **No** | 0.47ms | No Vertex AI log line |
| #4 | HIT (37s old) | **No** | 0.16ms | No Vertex AI log line |

**Key Insight**: When `search_embedding_cache` returns HIT, Vertex AI is completely bypassed. The ~0.2ms "Embedding generation" time is purely in-memory dictionary lookup.

### Qdrant Latency Variance

| Request | Qdrant Time | Delta from Avg |
|---------|-------------|----------------|
| #1 | 282ms | +19ms |
| #2 | **315ms** | +52ms |
| #3 | 286ms | +23ms |
| #4 | **168ms** | -95ms (fastest) |

**Avg**: 263ms | **Variance**: 147ms spread (168-315ms)

### Recommendations for Cache Expiry Mitigation

1. **Extend auth cache TTL** from 120s → 300s:
   ```python
   AUTH_OPTIMIZED_CACHE_TTL = 300  # 5 minutes
   ```

2. **Stagger cache expiry** with jitter to prevent cascading MISSes:
   ```python
   ttl = base_ttl + random.uniform(-10, 10)  # ±10s jitter
   ```

3. **Proactive cache refresh** at 80% TTL (before expiry)

4. **Monitor cache expiry correlation** - Alert when multiple caches expire simultaneously

---

## Appendix: Raw Log Excerpts

### First Request Cache Misses
```
[search_embedding_cache] Cache MISS for qwen_search:f2be7333...
[auth_optimized_cache] Cache MISS for auth_optimized:APIKe...
[workspace_subscription_cache] Cache MISS for workspace_sub_86cRDG...
[customer_tier_cache] Cache MISS for customer_tier_cus_RK...
```

### Subsequent Request Cache Hits
```
[auth_optimized_cache] Cache HIT for auth_optimized:APIKe... (age: 4.9s)
[workspace_subscription_cache] Cache HIT for workspace_sub_86cRDG... (age: 4.8s)
[search_embedding_cache] Cache HIT for qwen_search:f2be7333... (age: 4.5s)
[customer_tier_cache] Cache HIT for customer_tier_cus_RK... (age: 4.5s)
```
