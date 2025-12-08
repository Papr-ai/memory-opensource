# When to Use Temporal: Clarified Thresholds

## The Real Question: Throughput, Not User Count

The **10,000/day threshold** isn't about how many developers you have - it's about **total processing throughput**. A single developer doing bulk imports can hit this!

---

## Scenarios That Need Temporal

### ✅ Scenario 1: Bulk Document Import (Single Developer)

```
Developer uploads 5,000 PDFs in one batch
↓
Each PDF needs processing:
├── Extract text (30-60s per doc)
├── Chunk into memories (10-20s)
├── Generate embeddings (5-10s per chunk)
├── Index in vector DB (2-5s)
└── Create relationships (5-10s)

Total: 50-100 seconds per document
5,000 docs × 60s = 300,000 seconds = 83 hours of processing!
```

**Current approach**: ❌ Server would queue all 5,000 background tasks, run out of memory

**With Temporal**: ✅ Process 100 docs at a time across 10 workers, complete in 8 hours

### ✅ Scenario 2: Batch Memory Creation (API Integration)

```
Developer syncs their CRM (10,000 records)
↓
POST /v1/memories/batch with 10,000 memories
↓
Each memory needs:
├── Generate use cases (2-5s)
├── Find related memories (1-3s)
├── Generate schema (3-10s)
├── Index in Qdrant (1-2s)
└── Create Neo4j relationships (1-3s)

Total: 8-23 seconds per memory
10,000 memories × 15s = 150,000 seconds = 41 hours of processing!
```

**Current approach**: ❌ Would need to process 10k tasks, likely crash or timeout

**With Temporal**: ✅ Durable execution, resume on failures, webhook notification when done

### ✅ Scenario 3: Scheduled Sync (Daily Operations)

```
Developer has 1,000 users
Each user syncs 50 documents per day
= 50,000 documents/day to process
```

**Current approach**: ❌ Not designed for this scale

**With Temporal**: ✅ Built for this exact use case

---

## Updated Decision Matrix

| Your Situation | Use Current Approach | Use Temporal |
|----------------|---------------------|--------------|
| **Single memory creation** | ✅ Yes - Fast response | ❌ Overkill |
| **< 100 memories/request** | ✅ Yes - Works fine | 🤷 Optional |
| **100-1,000 memories/request** | ⚠️ Maybe - Will be slow | ✅ Recommended |
| **> 1,000 memories/request** | ❌ Will likely fail | ✅ Required |
| **Bulk PDF processing** | ❌ Not suitable | ✅ Required |
| **24/7 background sync** | ❌ Will lose state | ✅ Required |
| **Dev needs webhook notification** | ⚠️ Unreliable | ✅ Guaranteed |

---

## Recommended Hybrid Approach

### For Your Use Case

```python
# Route decision based on batch size
@router.post("/v1/memories/batch")
async def add_memory_batch(...):
    batch_size = len(memories)
    
    if batch_size <= 100:
        # Current approach: Fast, in-process
        return await process_with_background_tasks(memories)
    
    else:
        # Temporal: Durable, scalable
        return await process_with_temporal(memories)
```

### Implementation

```python
# config/features.py
class FeatureFlags:
    @property
    def use_temporal_for_batch(self) -> bool:
        """Use Temporal for batch operations above threshold"""
        return self.is_enabled("use_temporal_for_batch")
    
    @property
    def temporal_batch_threshold(self) -> int:
        """Batch size threshold to trigger Temporal"""
        return self.config.get("temporal", {}).get("batch_threshold", 100)

# routes/memory_routes.py
async def common_add_memory_batch_handler(...):
    features = get_features()
    batch_size = len(memories)
    
    if features.use_temporal_for_batch and batch_size > features.temporal_batch_threshold:
        logger.info(f"Using Temporal for batch of {batch_size} memories")
        return await process_batch_with_temporal(...)
    else:
        logger.info(f"Using background tasks for batch of {batch_size} memories")
        return await process_batch_with_background_tasks(...)
```

---

## Immediate Recommendation

Based on your requirements, you should:

### Phase 1: Now (This Week)
Add **batch size limits** to prevent overload:

```python
# .env
MAX_BATCH_SIZE=50  # Current: 50
MAX_CONCURRENT_PROCESSING=10  # Limit concurrent background tasks

# routes/memory_routes.py
if len(memories) > MAX_BATCH_SIZE:
    return BatchMemoryResponse.failure(
        error=f"Batch too large. Use Temporal API for batches > {MAX_BATCH_SIZE}",
        code=413
    )
```

### Phase 2: Next Month (If Needed)
Add **Temporal for large batches**:

```python
# New endpoint
@router.post("/v1/memories/batch/async")
async def add_memory_batch_async(...):
    """
    Async batch processing using Temporal.
    Returns immediately with batch_id.
    Sends webhook when complete.
    """
    if len(memories) < 100:
        return {"error": "Use /batch endpoint for small batches"}
    
    # Start Temporal workflow
    batch_id = await temporal_client.start_workflow(
        ProcessBatchWorkflow.run,
        memories,
        id=f"batch-{uuid.uuid4()}",
        task_queue="memory-batch-processing"
    )
    
    return {
        "batch_id": batch_id,
        "status": "processing",
        "webhook_url": webhook_url,
        "estimated_completion": "30-60 minutes"
    }
```

### Phase 3: When You Scale
Full Temporal integration for all async operations.

---

## Cost Analysis

### Current Approach Limitations

```
Server: 8 CPU cores, 16GB RAM
Max concurrent tasks: ~100-200 (memory limited)
Processing 1,000 memories: 41 minutes (serial)
Processing 10,000 memories: 7 hours (will crash)
```

### With Temporal

```
Temporal: 1 server
Workers: 10 servers × 8 cores = 80 concurrent tasks
Processing 1,000 memories: 5 minutes
Processing 10,000 memories: 50 minutes
Cost: +$500/mo infrastructure
```

### ROI

If a developer wants to process 10,000 memories:
- **Current**: They can't (or would take 7+ hours and crash)
- **With Temporal**: 50 minutes, guaranteed completion
- **Value**: Enables use cases you can't support today

---

## Bottom Line

**Start Temporal migration sooner** if you see any of these:

- [ ] Developers asking for bulk import (>100 items)
- [ ] Webhook notifications timing out
- [ ] Background tasks failing silently
- [ ] Batch operations taking >10 minutes
- [ ] Requests for "upload 1000 PDFs"

**For your case**: Since you mentioned "thousands of memories or docs", you'll likely need Temporal **within 3-6 months** of launch. Plan for it now, implement when you hit the first bulk use case.

---

## Quick Win: Rate Limiting

Until you have Temporal, protect your system:

```python
# Limit per developer
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/v1/memories/batch")
@limiter.limit("100/hour")  # 100 batch requests per hour
async def add_memory_batch(...):
    # Your existing code
    pass
```

This prevents one developer from overwhelming your system while you build Temporal support.

