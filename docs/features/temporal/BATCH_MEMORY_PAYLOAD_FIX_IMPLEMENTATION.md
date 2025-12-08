# Implementation Plan: Batch Memory Temporal Payload Fix

**Status**: Ready for Implementation
**Created**: 2025-10-21
**Last Updated**: 2025-10-21
**Estimated Effort**: 30-40 hours
**Timeline**: 3 weeks
**Related Documents**:
- [PRD](./BATCH_MEMORY_PAYLOAD_FIX_PRD.md)
- [Architecture](./BATCH_MEMORY_PAYLOAD_FIX_ARCHITECTURE.md)

---

## Table of Contents

1. [Overview](#overview)
2. [Implementation Phases](#implementation-phases)
3. [Detailed Task Breakdown](#detailed-task-breakdown)
4. [Deployment Strategy](#deployment-strategy)
5. [Testing Plan](#testing-plan)
6. [Rollback Procedures](#rollback-procedures)
7. [Success Metrics](#success-metrics)

---

## Overview

### Goals

Fix the critical bug where `add_memory_batch_v1` fails when processing large batches (50+ memories) due to Temporal gRPC payload limits by implementing Parse File storage for batch data.

### Key Changes

1. **Create BatchMemoryRequest Parse Class** - Store batch metadata and compressed file
2. **Refactor Temporal Workflow** - Accept only objectId reference instead of full payload
3. **Consolidate Activities** - Single activity fetches from Parse and processes batch
4. **Add Status Tracking** - Real-time progress via BatchMemoryRequest Parse object
5. **Remove Workaround** - Eliminate `build_shrunk_batch_data` complexity

### Expected Outcomes

- ✅ Support batches of 100+ memories (vs current ~50 limit)
- ✅ 99% reduction in workflow payload size (200KB → ~50 bytes)
- ✅ 60% reduction in workflow code complexity
- ✅ 99.9% success rate with proper error handling
- ✅ Zero breaking changes to API contract

---

## Implementation Phases

### Phase 1: Storage Layer (Week 1, Days 1-2)

**Goal**: Create Parse class and storage helpers

**Tasks**:
- Task 1.1: Create BatchMemoryRequest Parse class schema
- Task 1.2: Add Pydantic model for BatchMemoryRequest
- Task 1.3: Implement `create_batch_memory_request_in_parse()`
- Task 1.4: Implement `fetch_batch_memory_request_from_parse()`

**Effort**: 6-8 hours
**Priority**: High
**Dependencies**: None

### Phase 2: Workflow Layer (Week 1, Days 3-4)

**Goal**: Refactor Temporal workflow and activities

**Tasks**:
- Task 2.1: Create new simplified workflow
- Task 2.2: Implement `fetch_and_process_batch_memories()` activity
- Task 2.3: Remove `build_shrunk_batch_data` workaround

**Effort**: 8-10 hours
**Priority**: High
**Dependencies**: Phase 1 complete

### Phase 3: API Layer (Week 1, Day 5)

**Goal**: Update API endpoint to use new storage pattern

**Tasks**:
- Task 3.1: Modify `process_batch_with_temporal()` to always use Parse
- Task 3.2: Update workflow starter to use new workflow

**Effort**: 4-6 hours
**Priority**: High
**Dependencies**: Phase 2 complete

### Phase 4: Observability (Week 2, Days 1-2)

**Goal**: Add status tracking and monitoring

**Tasks**:
- Task 4.1: Add status update helpers
- Task 4.2: Add progress query endpoint (`GET /v1/batch/{batch_id}/status`)

**Effort**: 4-6 hours
**Priority**: Medium
**Dependencies**: Phase 3 complete

### Phase 5: Testing (Week 2, Days 3-5)

**Goal**: Comprehensive testing and validation

**Tasks**:
- Task 5.1: Unit tests for storage layer
- Task 5.2: Integration tests for workflow
- Task 5.3: Load testing and migration validation

**Effort**: 8-10 hours
**Priority**: High
**Dependencies**: Phases 1-4 complete

---

## Detailed Task Breakdown

### Phase 1: Storage Layer

#### Task 1.1: Create BatchMemoryRequest Parse Class

**Complexity**: Moderate
**Effort**: 2 hours
**Files**: Parse Server dashboard or migration script

**Steps**:

1. Create Parse Server class schema:
```javascript
{
  className: "BatchMemoryRequest",
  fields: {
    batchId: String,
    requestId: String,
    organization: Pointer<Organization>,
    namespace: Pointer<Namespace>,
    user: Pointer<_User>,
    workspace: Pointer<WorkSpace>,
    batchDataFile: File,
    batchMetadata: Object,
    status: String,
    processedCount: Number,
    successCount: Number,
    failCount: Number,
    workflowId: String,
    workflowRunId: String,
    webhookUrl: String,
    webhookSecret: String,
    webhookSent: Boolean,
    startedAt: Date,
    completedAt: Date,
    processingDurationMs: Number,
    errors: Array
  }
}
```

2. Create indexes:
```javascript
db.BatchMemoryRequest.createIndex({ batchId: 1 })
db.BatchMemoryRequest.createIndex({ organization: 1, namespace: 1 })
db.BatchMemoryRequest.createIndex({ user: 1, status: 1 })
db.BatchMemoryRequest.createIndex({ status: 1, createdAt: 1 })
```

3. Set class-level permissions:
```javascript
{
  find: { requiresAuthentication: true },
  get: { requiresAuthentication: true },
  create: { requiresAuthentication: true },
  update: { requiresAuthentication: true },
  delete: { requiresAuthentication: true }
}
```

**Acceptance Criteria**:
- [ ] BatchMemoryRequest class created with all fields
- [ ] All indexes created successfully
- [ ] Class-level permissions configured
- [ ] Can create/read/update via REST API
- [ ] Test record created and fetched successfully

**Testing**:
```bash
# Test creating a batch request
curl -X POST \
  -H "X-Parse-Application-Id: ${APP_ID}" \
  -H "X-Parse-REST-API-Key: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "batchId": "test-batch-001",
    "status": "pending",
    "totalMemories": 10
  }' \
  https://parse-server-url/parse/classes/BatchMemoryRequest
```

---

#### Task 1.2: Add Pydantic Model

**Complexity**: Simple
**Effort**: 1-2 hours
**Files**: `models/parse_server.py`

**Steps**:

1. Add import statements (if needed):
```python
from typing import Optional, Dict, List, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
```

2. Add model after `PostParseServer` class (around line 1940):
```python
class BatchMemoryRequest(BaseModel):
    """Model for BatchMemoryRequest class in Parse Server."""
    objectId: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    ACL: Dict[str, Dict[str, bool]] = Field(default_factory=dict)
    className: Optional[str] = Field(default="BatchMemoryRequest")

    # [Full model definition from architecture doc]
```

3. Add serialization method:
```python
def model_dump(self, *args, **kwargs):
    """Override to transform pointers to __type format"""
    # [Implementation from architecture doc]
```

**Acceptance Criteria**:
- [ ] Model validates correctly with test data
- [ ] Pointer fields serialize with `__type`
- [ ] File field serializes correctly
- [ ] Can round-trip to/from Parse JSON
- [ ] All optional fields work correctly

**Testing**:
```python
# Test model validation
test_data = {
    "batchId": "test-001",
    "status": "pending",
    "processedCount": 0
}

model = BatchMemoryRequest(**test_data)
assert model.batchId == "test-001"
assert model.status == "pending"

# Test serialization
dumped = model.model_dump()
assert "__type" in dumped.get("organization", {})
```

---

#### Task 1.3: Create Storage Helper

**Complexity**: Moderate
**Effort**: 2-3 hours
**Files**: `services/memory_management.py`

**Location**: Add after `store_batch_memories_in_parse()` (around line 4887)

**Implementation**: See full function in Architecture doc

**Key Points**:
- Compress batch data with gzip (target 7-10x compression)
- Create Parse File with compressed data
- Set proper ACL (user + organization)
- Return objectId of created request
- Log compression metrics

**Acceptance Criteria**:
- [ ] Creates BatchMemoryRequest in Parse successfully
- [ ] Compresses batch data (>5x ratio achieved)
- [ ] Uploads as Parse File
- [ ] Sets proper ACL (user + org read)
- [ ] Returns objectId
- [ ] Handles errors gracefully (network, Parse errors)
- [ ] Logs creation with metrics

**Testing**:
```python
async def test_create_batch_request():
    memories = [{"content": f"Memory {i}"} for i in range(100)]

    request_id = await create_batch_memory_request_in_parse(
        batch_request=BatchMemoryRequest(memories=memories),
        auth_response=test_auth,
        batch_id="test-batch-001"
    )

    assert request_id is not None
    assert len(request_id) > 0

    # Verify in Parse
    batch_req = await fetch_from_parse("BatchMemoryRequest", request_id)
    assert batch_req["status"] == "pending"
    assert batch_req["batchDataFile"]["__type"] == "File"
```

---

#### Task 1.4: Create Fetch Helper

**Complexity**: Moderate
**Effort**: 2-3 hours
**Files**: `services/memory_management.py`

**Location**: Add after Task 1.3 function

**Implementation**: See full function in Architecture doc

**Key Points**:
- Fetch BatchMemoryRequest by objectId
- Download batchDataFile from URL
- Decompress with gzip
- Parse JSON back to memories
- Handle missing files gracefully

**Acceptance Criteria**:
- [ ] Fetches BatchMemoryRequest by objectId
- [ ] Downloads file from URL successfully
- [ ] Decompresses data correctly
- [ ] Parses JSON back to memories list
- [ ] Returns None if not found (404)
- [ ] Handles network errors gracefully
- [ ] Logs fetch with metrics

**Testing**:
```python
async def test_fetch_batch_request():
    # Create first
    request_id = await create_batch_memory_request_in_parse(...)

    # Fetch back
    batch_req = await fetch_batch_memory_request_from_parse(request_id)

    assert batch_req is not None
    assert batch_req.objectId == request_id
    assert len(batch_req.memories) == 100
    assert batch_req.status == "pending"
```

---

### Phase 2: Workflow Layer

#### Task 2.1: Create New Workflow

**Complexity**: Simple
**Effort**: 2-3 hours
**Files**: `cloud_plugins/temporal/workflows/batch_memory.py`

**Steps**:

1. Add new workflow class:
```python
@workflow.defn
class ProcessBatchMemoryFromRequestWorkflow:
    """Simplified workflow using Parse storage."""

    @workflow.run
    async def run(self, request_ref: Dict[str, str]) -> Dict[str, Any]:
        # [Implementation from architecture doc]
```

2. Keep old workflow for backwards compatibility:
```python
# Keep existing ProcessBatchMemoryWorkflow for migration period
# Will be removed after 1 week
```

**Acceptance Criteria**:
- [ ] New workflow accepts minimal payload (<100 bytes)
- [ ] Workflow calls single activity
- [ ] Proper retry policy configured
- [ ] Logging at start and completion
- [ ] Old workflow still works (no changes)

**Testing**:
```python
async def test_new_workflow():
    request_ref = {
        "request_id": "test-req-001",
        "batch_id": "test-batch-001"
    }

    result = await worker.execute_workflow(
        ProcessBatchMemoryFromRequestWorkflow.run,
        request_ref
    )

    assert result["status"] == "completed"
    assert result["total"] > 0
```

---

#### Task 2.2: Create New Activity

**Complexity**: Complex
**Effort**: 4-5 hours
**Files**: `cloud_plugins/temporal/activities/memory_activities.py`

**Location**: Add after `process_batch_memories_from_parse_reference()` (around line 806)

**Implementation**: See full function in Architecture doc

**Key Points**:
- Fetch BatchMemoryRequest from Parse
- Update status to "processing"
- Loop through memories and process each
- Update progress every 10 memories
- Send heartbeat every 10 memories
- Update final status and send webhook (if configured)
- Handle partial failures gracefully

**Acceptance Criteria**:
- [ ] Activity fetches batch from Parse successfully
- [ ] Processes all memories through pipeline
- [ ] Updates status after each stage
- [ ] Sends heartbeat periodically
- [ ] Stores errors in BatchMemoryRequest
- [ ] Handles partial failures (some succeed, some fail)
- [ ] Is idempotent (can retry safely)
- [ ] Uses existing webhook infrastructure

**Testing**:
```python
async def test_fetch_and_process_activity():
    # Setup: Create batch request
    request_id = await create_batch_memory_request_in_parse(...)

    # Execute activity
    result = await fetch_and_process_batch_memories(
        request_id=request_id,
        batch_id="test-batch"
    )

    assert result["status"] == "completed"
    assert result["successful"] == 100
    assert result["failed"] == 0

    # Verify batch request updated
    batch_req = await fetch_batch_memory_request_from_parse(request_id)
    assert batch_req.status == "completed"
    assert batch_req.processedCount == 100
```

---

#### Task 2.3: Remove Workaround Code

**Complexity**: Simple
**Effort**: 1-2 hours
**Files**: `cloud_plugins/temporal/workflows/batch_memory.py`

**Steps**:

1. Delete `build_shrunk_batch_data()` function (lines 76-134)
2. Update `ProcessBatchMemoryWorkflow` to use new pattern (optional, for migration)
3. Update comments/documentation
4. Remove any references to shrinking logic

**Acceptance Criteria**:
- [ ] Workflow code reduced significantly
- [ ] No per-memory payload construction
- [ ] Cleaner, more maintainable code
- [ ] All tests still pass
- [ ] Documentation updated

---

### Phase 3: API Layer

#### Task 3.1: Modify Batch Processor

**Complexity**: Moderate
**Effort**: 2-3 hours
**Files**: `services/batch_processor.py`

**Changes**:

Remove threshold logic (lines 109-129) and always use Parse:

```python
# Before:
large_batch_threshold = int(os.getenv("TEMPORAL_REFERENCE_BATCH_THRESHOLD", "40"))
if len(batch_request.memories) >= large_batch_threshold:
    # Use Parse storage
else:
    # Use direct payload

# After:
# Always create BatchMemoryRequest in Parse
request_id = await create_batch_memory_request_in_parse(...)
workflow_id = await start_batch_workflow_with_reference(...)
```

**Acceptance Criteria**:
- [ ] All batches use Parse storage (no threshold)
- [ ] Creates BatchMemoryRequest before starting workflow
- [ ] Passes only request_id to workflow
- [ ] Returns batch_id and workflow_id to client
- [ ] Backwards compatible response format
- [ ] Existing webhook infrastructure used

**Testing**:
```python
async def test_process_batch_always_uses_parse():
    # Small batch
    small_batch = BatchMemoryRequest(memories=[...])  # 10 memories
    result = await process_batch_with_temporal(small_batch, ...)
    # Should still use Parse storage

    # Large batch
    large_batch = BatchMemoryRequest(memories=[...])  # 150 memories
    result = await process_batch_with_temporal(large_batch, ...)
    # Should use Parse storage
```

---

#### Task 3.2: Update Workflow Starter

**Complexity**: Simple
**Effort**: 1-2 hours
**Files**: `cloud_plugins/temporal/workflows/batch_memory.py`

**Add helper function**:

```python
async def start_batch_workflow_with_reference(
    client: Any,
    request_id: str,
    batch_id: str
) -> str:
    """Start batch workflow with Parse reference only."""
    workflow_id = f"batch-{batch_id}"

    handle = await client.start_workflow(
        ProcessBatchMemoryFromRequestWorkflow.run,
        {"request_id": request_id, "batch_id": batch_id},
        id=workflow_id,
        task_queue=task_queue
    )

    return workflow_id
```

**Acceptance Criteria**:
- [ ] Workflow started with <100 byte payload
- [ ] Can track workflow by ID
- [ ] Returns immediately with workflow_id
- [ ] Proper task queue configuration

---

### Phase 4: Observability

#### Task 4.1: Add Status Update Helpers

**Complexity**: Moderate
**Effort**: 2-3 hours
**Files**: `services/memory_management.py`

**Add functions**:

1. `update_batch_request_status()` - Update status field
2. `update_batch_request_progress()` - Update processedCount
3. `update_batch_request_completion()` - Finalize with results
4. `mark_webhook_sent()` - Mark webhook as delivered

**Acceptance Criteria**:
- [ ] Updates status atomically
- [ ] Tracks timing metadata
- [ ] Stores error details
- [ ] Calculates progress percentage
- [ ] Returns success/failure

---

#### Task 4.2: Add Progress Query Endpoint

**Complexity**: Moderate
**Effort**: 2-3 hours
**Files**: `routers/v1/memory_routes_v1.py`

**Add endpoint**:

```python
@router.get("/batch/{batch_id}/status")
async def get_batch_status(
    batch_id: str,
    user: User = Depends(get_current_user),
    organization: Organization = Depends(get_current_organization)
):
    """Get real-time batch processing status."""
    # [Implementation]
```

**Response**:
```json
{
  "batch_id": "batch-abc123",
  "status": "processing",
  "total_memories": 150,
  "processed": 87,
  "successful": 85,
  "failed": 2,
  "progress_percent": 58.0,
  "started_at": "2025-10-21T10:00:00Z",
  "estimated_completion": "2025-10-21T10:15:00Z",
  "errors": [...]
}
```

**Acceptance Criteria**:
- [ ] Fetches BatchMemoryRequest by batchId
- [ ] Returns real-time progress
- [ ] Estimates completion time
- [ ] Shows error details
- [ ] Requires authentication
- [ ] Respects ACL (only creator can view)

---

### Phase 5: Testing

#### Task 5.1: Unit Tests

**Complexity**: Moderate
**Effort**: 3-4 hours
**Files**: `tests/test_batch_memory_storage.py` (new)

**Tests to add**:

```python
# Storage tests
async def test_create_batch_request_in_parse()
async def test_fetch_batch_request_from_parse()
async def test_update_batch_request_status()
async def test_batch_request_compression_ratio()
async def test_batch_request_acl()
async def test_batch_request_error_storage()

# Edge cases
async def test_fetch_nonexistent_request()
async def test_create_with_invalid_data()
async def test_large_batch_storage()
```

**Acceptance Criteria**:
- [ ] All tests pass
- [ ] Code coverage >80%
- [ ] Tests use real Parse Server (test env)
- [ ] Tests clean up after themselves

---

#### Task 5.2: Integration Tests

**Complexity**: Complex
**Effort**: 4-5 hours
**Files**: `tests/test_temporal_batch_processing.py`

**Tests to add**:

```python
async def test_batch_workflow_with_parse_reference()
async def test_large_batch_processing()  # 100+ memories
async def test_batch_failure_recovery()
async def test_batch_status_tracking()
async def test_batch_webhook_delivery()
async def test_concurrent_batches()
```

**Acceptance Criteria**:
- [ ] All tests pass
- [ ] Works with real Temporal worker
- [ ] Parse storage verified
- [ ] Webhook delivery tested
- [ ] Can run in CI/CD pipeline

---

#### Task 5.3: Load Testing and Migration

**Complexity**: Complex
**Effort**: 2-3 hours

**Load test scenarios**:

1. **Concurrent Batches**: 10 concurrent requests of 100 memories each
2. **Large Batch**: Single batch of 500 memories
3. **Sustained Load**: 1 batch per minute for 1 hour

**Acceptance Criteria**:
- [ ] All scenarios complete successfully
- [ ] No memory leaks
- [ ] No Parse storage quota exceeded
- [ ] Performance within acceptable range
- [ ] Migration documentation complete

---

## Deployment Strategy

### Pre-Deployment Checklist

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Parse Server schema updated
- [ ] Feature flag implemented (`USE_PARSE_BATCH_STORAGE`)
- [ ] Monitoring dashboards created
- [ ] Rollback procedure documented

### Deployment Timeline

#### Day 1: Deploy with Feature Flag Disabled

```bash
# Deploy code to production
USE_PARSE_BATCH_STORAGE=false
```

**Actions**:
- Deploy code with flag off
- Verify old workflow still works
- Monitor for any regressions

**Success Criteria**:
- No errors in logs
- Batch processing continues normally
- No user complaints

---

#### Day 2: Enable in Staging

```bash
# Staging environment
USE_PARSE_BATCH_STORAGE=true
```

**Actions**:
- Enable flag in staging
- Run full test suite
- Process 100+ real batches
- Verify Parse storage working

**Success Criteria**:
- All tests pass
- Batches process successfully
- Parse files created and deleted properly

---

#### Day 3: Canary Release (10%)

```python
# Canary logic (example)
if hash(user_id) % 10 == 0:
    USE_PARSE_BATCH_STORAGE = true
```

**Actions**:
- Enable for 10% of users
- Monitor for 6 hours
- Track metrics closely

**Metrics to watch**:
- Batch success rate
- Parse storage usage
- Workflow completion time
- Error rates

**Success Criteria**:
- Success rate ≥ baseline
- No Parse errors
- No workflow timeouts

---

#### Days 4-7: Gradual Rollout

- **Day 4**: 25% of traffic
- **Day 5**: 50% of traffic
- **Day 6**: 75% of traffic
- **Day 7**: 100% of traffic

**Actions at each step**:
- Increase percentage
- Monitor for 4-6 hours
- Check metrics
- Verify no issues

---

#### Day 14: Cleanup

**Actions**:
- Verify all in-flight old workflows completed
- Remove old workflow code
- Remove `build_shrunk_batch_data` workaround
- Remove feature flag
- Update documentation

---

## Rollback Procedures

### If Issues Detected

**Step 1: Immediate Rollback**
```bash
# Set feature flag to false
USE_PARSE_BATCH_STORAGE=false
```

Takes effect in <30 seconds for new requests.

**Step 2: Monitor Recovery**
- Watch for old workflow usage increasing
- Verify batch processing returns to normal
- Check error rates normalize

**Step 3: Investigate**
- Review Parse/Temporal logs
- Identify root cause
- Fix in staging before re-attempting

### If Parse Server Issues

**Symptoms**:
- Parse Server health degraded
- File storage quota exceeded
- CDN download failures

**Actions**:
1. Check Parse Server metrics
2. Verify file storage quota
3. Check CDN health
4. If Parse unavailable, rollback immediately

---

## Success Metrics

### Primary Metrics

1. **Workflow History Size**
   - Before: ~500KB per batch (100 memories)
   - After: <10KB per batch
   - Target: >95% reduction ✅

2. **Activity Payload Size**
   - Before: ~450KB total for 100 memories
   - After: ~50 bytes (request_id reference)
   - Target: >99% reduction ✅

3. **Batch Processing Time**
   - Before: ~3-5 minutes for 100 memories
   - After: ~2-4 minutes
   - Target: No regression, 10-20% improvement

4. **Success Rate**
   - Before: ~98% (baseline)
   - After: ≥98%
   - Target: No regression ✅

### Secondary Metrics

5. **Parse Storage Usage**
   - Expected: ~50KB per batch × N batches/day
   - Monitor: Should not exceed quota
   - Target: <1GB/day

6. **Compression Ratio**
   - Expected: 7-8x for JSON
   - Monitor: Actual ratio achieved
   - Target: >5x consistently

7. **Code Maintainability**
   - Before: 265 lines in workflow
   - After: <100 lines
   - Target: >60% reduction ✅

---

## Risk Mitigation

### Risk 1: Parse Server Load

**Mitigation**:
- Compression reduces file size by 10x
- Parse Server can handle thousands of file ops/sec
- CDN caches downloads
- Monitor Parse metrics closely

### Risk 2: Migration Disruption

**Mitigation**:
- Keep old workflow for compatibility
- Feature flag for controlled rollout
- Old workflows complete gracefully
- Gradual percentage increase

### Risk 3: Data Loss

**Mitigation**:
- Activity is idempotent
- Store progress in BatchMemoryRequest
- Can resume from last checkpoint
- Temporal retries automatically

---

## Post-Implementation

### Week 1 After Launch

- [ ] Monitor production metrics daily
- [ ] Review error logs
- [ ] Gather user feedback
- [ ] Document any issues

### Week 2 After Launch

- [ ] Analyze performance data
- [ ] Optimize if needed
- [ ] Update documentation
- [ ] Share lessons learned

### Month 1 After Launch

- [ ] Review success metrics
- [ ] Plan future enhancements
- [ ] Archive old workflow code
- [ ] Celebrate success! 🎉

---

## Appendix

### Useful Commands

**Check batch request status**:
```bash
curl -X GET \
  -H "X-Parse-Application-Id: ${APP_ID}" \
  https://parse-server/parse/classes/BatchMemoryRequest/{objectId}
```

**Monitor Temporal workflows**:
```bash
temporal workflow list --query 'WorkflowType="ProcessBatchMemoryFromRequestWorkflow"'
```

**Check Parse storage usage**:
```bash
# Query total file storage
db.BatchMemoryRequest.aggregate([
  { $group: { _id: null, totalSize: { $sum: "$batchMetadata.compressed_size_bytes" } } }
])
```

### Contact Information

- **Engineering Lead**: [Name]
- **On-Call Engineer**: [Name]
- **Parse Server Admin**: [Name]
- **Temporal Admin**: [Name]

---

**Document Version**: 1.0
**Last Review**: 2025-10-21
**Next Review**: After implementation complete
