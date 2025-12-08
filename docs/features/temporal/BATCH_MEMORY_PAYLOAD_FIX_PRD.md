# PRD: Batch Memory Processing Payload Size Fix

**Status**: Draft
**Created**: 2025-10-21
**Last Updated**: 2025-10-21
**Owner**: Engineering Team
**Priority**: High

---

## Executive Summary

The memory server's `add_memory_batch_v1` endpoint currently fails when processing large batches (50+ memories) due to gRPC payload size limitations when data is passed to Temporal workflows. This PRD outlines a solution that follows existing codebase patterns for handling large payloads through Parse File storage, ensuring scalability to 100+ memories per batch while maintaining backwards compatibility.

**Business Impact:**
- **Current State**: Users cannot reliably process batches larger than ~50 memories
- **Target State**: Support batches of 100+ memories with guaranteed reliability
- **User Benefit**: Enables bulk memory operations for power users and enterprise customers
- **Technical Debt**: Aligns batch processing with established patterns used elsewhere in the codebase

---

## Problem Statement

### Current Pain Points

1. **Size Limitations**: The `add_memory_batch_v1` endpoint passes the complete batch payload directly to Temporal workflows, causing failures when:
   - Batch contains 50+ memories with substantial content
   - Individual memories contain large text or metadata
   - Combined payload exceeds Temporal's default 2MB gRPC limit

2. **Inconsistent Patterns**: Current batch processing differs from the established pattern used in `memory_management.py`:
   - `store_extraction_result_in_post` stores large data in Parse Files
   - `fetch_post_with_provider_result_async` retrieves it asynchronously
   - Batch processing sends everything through workflow arguments

3. **User Impact**:
   - Unpredictable failures without clear error messages
   - No guidance on maximum batch size
   - Forces users to manually split batches without clear size limits

### Root Cause Analysis

**File**: `cloud_plugins/temporal/workflows/batch_memory.py`

**Current Problematic Pattern**:
```python
def build_shrunk_batch_data(idx: int) -> Dict[str, Any]:
    # Extracts single memory as plain dict
    # Creates slim auth response
    # Returns shrunk payload with only ONE memory
    # This is sent per-activity to avoid gRPC limits
```

**Issues**:
1. **Complexity**: Creates individual payloads for each memory in the batch
2. **Scalability**: Still sends N payloads for N memories (reduced size, but still N network calls)
3. **Maintenance**: Complex workaround code that's hard to understand
4. **Not DRY**: Doesn't reuse existing Parse storage infrastructure

### Business Impact

- **User Experience**: Unreliable batch operations lead to user frustration
- **Scalability**: Limits enterprise adoption for bulk data ingestion
- **Support Burden**: Increases support tickets for "mysterious" batch failures
- **Technical Debt**: Inconsistent patterns make codebase harder to maintain

---

## Success Metrics

### Key Performance Indicators

1. **Batch Size Capacity**
   - Target: Successfully process batches of 100+ memories
   - Measurement: End-to-end batch processing without gRPC errors
   - Baseline: Current ~50 memory limit

2. **Reliability**
   - Target: 99.9% success rate for valid batches
   - Measurement: Error rate tracking in production
   - Baseline: Current unreliable above 50 memories

3. **Performance**
   - Target: < 5 second overhead for Parse File storage/retrieval
   - Measurement: Time delta between old and new implementation
   - Baseline: Current direct payload passing (no storage overhead)

4. **Backwards Compatibility**
   - Target: Zero breaking changes to API contract
   - Measurement: All existing API tests pass unchanged
   - Success Criteria: No client updates required

### Acceptance Criteria

**Must Have:**
- ✅ Process batches of 100+ memories without gRPC errors
- ✅ Maintain exact API response format
- ✅ Follow established Parse File storage pattern from `memory_management.py`
- ✅ Pass all existing test suites without modification
- ✅ Proper error handling and logging at each stage
- ✅ Secure access control for stored batch data

**Should Have:**
- ✅ Clear documentation of maximum practical batch size
- ✅ Informative error messages if size limits exceeded
- ✅ Metrics/logging for batch processing performance
- ✅ Recovery mechanism using Temporal's durable execution

**Could Have:**
- 🔄 Automatic batch splitting for very large requests
- 🔄 Progress tracking for long-running batches
- 🔄 Webhook callbacks for batch completion

---

## User Stories

### Primary User Flow: Large Batch Processing

**As a** power user ingesting bulk historical data
**I want to** submit batches of 100+ memories in a single API call
**So that** I can efficiently populate my memory space without manual batching

**Acceptance Criteria:**
```python
# User submits large batch
response = requests.post(
    "/api/v1/memory/batch",
    json={
        "memories": [
            {"content": "...", "metadata": {...}},
            # ... 100+ memories
        ],
        "namespace": "historical-import"
    }
)

# Should succeed with:
assert response.status_code == 200
assert response.json()["success"] == True
assert len(response.json()["results"]) == len(request_memories)
```

### Edge Case: Extremely Large Individual Memories

**As a** system administrator
**I want** clear error messages when individual memories exceed reasonable size
**So that** I can identify and correct data quality issues

**Acceptance Criteria:**
- Individual memory > 10MB: Clear error message with size information
- Batch file > 50MB: Warning logged, potentially split processing
- Invalid data structure: Detailed validation error before storage

### Error Recovery Flow

**As an** operations engineer
**I want** batch processing to recover from server crashes
**So that** partially processed batches don't leave inconsistent state

**Acceptance Criteria:**
- Temporal workflow can resume from last completed activity
- Parse File storage persists across workflow retries
- Clear audit trail of which memories were successfully processed

---

## Technical Specifications

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                       │
│                  memory_routes_v1.py                         │
└────────────────────────────┬────────────────────────────────┘
                             │
                             │ 1. Create BatchMemoryRequest
                             │    with Parse File storage
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  Parse Server Storage                        │
│                                                               │
│  BatchMemoryRequest (class)                                  │
│  ├── organization: Pointer                                   │
│  ├── user: Pointer                                           │
│  ├── workspace: Pointer                                      │
│  ├── namespace: Pointer                                      │
│  ├── batchFile: File (stores full memories JSON)            │
│  └── status: String (pending/processing/completed/failed)   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             │ 2. Pass only objectId to workflow
                             │    (small payload)
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Temporal Workflow (batch_memory.py)             │
│                                                               │
│  Input: batch_request_id (String, ~20 bytes)                │
│                                                               │
│  Steps:                                                       │
│  1. Call fetch_batch_request activity                        │
│  2. Call process_memories activity (chunked)                 │
│  3. Call update_batch_status activity                        │
└────────────────────────────┬────────────────────────────────┘
                             │
                             │ 3. Activities fetch data,
                             │    process memories,
                             │    update status
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│         Temporal Activities (memory_activities.py)           │
│                                                               │
│  fetch_batch_request_activity(batch_request_id)             │
│  ├── Fetch BatchMemoryRequest from Parse                    │
│  ├── Download and parse batchFile                           │
│  └── Return memories list                                    │
│                                                               │
│  process_memories_activity(memories, context)               │
│  ├── Process each memory with existing logic                │
│  ├── Return results with success/failure status             │
│  └── Handle partial failures gracefully                     │
│                                                               │
│  update_batch_status_activity(batch_request_id, status)    │
│  ├── Update BatchMemoryRequest status                       │
│  └── Store final results                                     │
└─────────────────────────────────────────────────────────────┘
```

### Data Models

#### New Parse Class: BatchMemoryRequest

```python
# Location: cloud_plugins/parse/schema/batch_memory_request.py

from parse_rest.datatypes import Object, Pointer, File

class BatchMemoryRequest(Object):
    """
    Stores batch memory request data in Parse for processing.

    This class follows the pattern established in memory_management.py
    for storing large payloads in Parse Files to avoid gRPC size limits.
    """

    # Pointers to related entities
    organization: Pointer  # Required: Organization context
    user: Pointer          # Required: User who submitted batch
    workspace: Pointer     # Optional: Workspace context
    namespace: Pointer     # Optional: Namespace for memories

    # Large payload storage
    batchFile: File        # Required: Parse File containing JSON array of memories

    # Status tracking
    status: str            # pending | processing | completed | failed
    error: str             # Optional: Error message if failed

    # Metadata
    totalMemories: int     # Total number of memories in batch
    processedMemories: int # Number successfully processed

    # Audit fields
    createdAt: datetime    # Auto-populated by Parse
    updatedAt: datetime    # Auto-populated by Parse

    # Results storage
    resultsFile: File      # Optional: Parse File with detailed results
```

#### Memory Data Structure (unchanged)

```python
# Existing structure - no changes needed
class MemoryItem(TypedDict):
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]]
    source: Optional[str]
    tags: Optional[List[str]]
```

### API Design

#### Endpoint: POST /api/v1/memory/batch (unchanged)

**Request Schema:**
```python
class BatchMemoryRequest(BaseModel):
    memories: List[MemoryItem]  # 1 to 100+ items
    namespace: Optional[str] = None
    workspace_id: Optional[str] = None
    # ... other existing fields
```

**Response Schema (unchanged):**
```python
class BatchMemoryResponse(BaseModel):
    success: bool
    batch_id: str  # New: BatchMemoryRequest objectId for tracking
    total: int
    processed: int
    failed: int
    results: List[MemoryProcessingResult]
    errors: Optional[List[str]]
```

---

## Testing Requirements

### Unit Tests

**Test File**: `tests/services/test_batch_memory_storage.py`

**Test Coverage**:
- `test_store_batch_request_success()` - Batch request creation
- `test_fetch_batch_request_data_success()` - Batch data retrieval
- `test_fetch_batch_request_access_denied()` - Access control
- `test_store_large_batch()` - Maximum size handling
- `test_batch_request_compression_ratio()` - Compression effectiveness

**Acceptance Criteria**:
- All tests pass
- Code coverage >80%
- Tests use real Parse Server (test environment)

### Integration Tests

**Test File**: `tests/workflows/test_batch_memory_workflow.py`

**Test Coverage**:
- `test_workflow_processes_batch_successfully()` - End-to-end workflow
- `test_workflow_handles_partial_failures()` - Error resilience
- `test_workflow_survives_activity_retries()` - Durable execution
- `test_batch_api_large_batch()` - 100+ memory processing

**Acceptance Criteria**:
- All tests pass
- Works with real Temporal worker
- Parse storage verified

### Load Testing

**Test Scenarios**:

1. **Concurrent Batches**
   - 10 concurrent batch requests of 100 memories each
   - Measure: Success rate, average latency, p95 latency
   - Target: 100% success, < 60s average, < 120s p95

2. **Large Individual Batches**
   - Single batch of 500 memories
   - Measure: Processing time, memory usage, success rate
   - Target: Complete in < 5 minutes, < 500MB memory

3. **Sustained Load**
   - 1 batch per minute for 1 hour
   - Measure: Success rate, system stability, resource usage
   - Target: 100% success, stable resource usage

---

## Timeline and Milestones

### Phase 1: Core Implementation (Week 1)

**Day 1-2: Storage Layer**
- [ ] Create BatchMemoryRequest Parse class schema
- [ ] Implement `store_batch_request` helper
- [ ] Implement `fetch_batch_request_data` helper
- [ ] Write unit tests for storage helpers

**Day 3-4: Workflow Layer**
- [ ] Update BatchMemoryWorkflow to use batch_request_id
- [ ] Implement new Temporal activities
- [ ] Add chunked processing logic
- [ ] Write workflow unit tests

**Day 5: API Layer**
- [ ] Update add_memory_batch_v1 endpoint
- [ ] Add validation and error handling
- [ ] Write API integration tests
- [ ] Update API documentation

### Phase 2: Testing and Validation (Week 2)

**Day 1-2: Integration Testing**
- [ ] Run full integration test suite
- [ ] Test with various batch sizes (10, 50, 100, 150)
- [ ] Verify Parse File storage and retrieval
- [ ] Test error scenarios and recovery

**Day 3: Load Testing**
- [ ] Execute concurrent batch load tests
- [ ] Test sustained load scenarios
- [ ] Profile resource usage and performance
- [ ] Document performance benchmarks

**Day 4: Edge Case Testing**
- [ ] Test all identified edge cases
- [ ] Verify Temporal retry logic
- [ ] Test failure recovery scenarios
- [ ] Document any limitations found

**Day 5: Documentation**
- [ ] Update API documentation
- [ ] Write operational runbook
- [ ] Document monitoring and alerting
- [ ] Create troubleshooting guide

### Phase 3: Deployment (Week 3)

**Day 1: Staging Deployment**
- [ ] Deploy to staging environment
- [ ] Run smoke tests
- [ ] Verify Temporal workflows execute
- [ ] Test with production-like data

**Day 2-3: Production Preparation**
- [ ] Set up monitoring dashboards
- [ ] Configure alerts for failures
- [ ] Prepare rollback plan
- [ ] Brief support team

**Day 4: Production Deployment**
- [ ] Deploy during low-traffic window
- [ ] Monitor initial batches closely
- [ ] Verify metrics and logs
- [ ] Gradual rollout to all users

**Day 5: Post-Launch**
- [ ] Monitor production metrics
- [ ] Gather user feedback
- [ ] Address any issues
- [ ] Document lessons learned

---

## Risk Assessment

### Risk 1: Parse Server Load

**Impact**: Medium
**Probability**: Low
**Mitigation**:
- Compression reduces file size by 10x
- Parse Server can handle thousands of file ops/sec
- CDN caches file downloads
- Monitor Parse Server metrics

### Risk 2: Migration Disruption

**Impact**: Medium
**Probability**: Low
**Mitigation**:
- Keep old workflow alongside new one
- Use feature flag to control which workflow is started
- Old workflows complete gracefully
- Remove old code after 7 days

### Risk 3: Batch Data Loss

**Impact**: High
**Probability**: Low
**Mitigation**:
- Activity is idempotent (can retry safely)
- Store `processedCount` after each memory
- On retry, skip already-processed memories
- Final status has exact success/fail counts

---

## Future Considerations

### Scalability Enhancements

1. **Parallel Chunk Processing**
   - Process multiple chunks concurrently
   - Potential 2-3x speedup for large batches

2. **Batch Splitting API**
   - Automatic splitting for very large batches
   - Client-side chunking helper library

3. **Streaming Results**
   - WebSocket/SSE for progress updates
   - Real-time feedback for long batches

### Monitoring and Observability

**Metrics to Track**:
- Batch size distribution
- Processing time by batch size
- Failure rates and reasons
- Parse File storage usage
- Temporal workflow durations

**Alerting Rules**:
- Batch failure rate > 5%
- Average processing time > 2 minutes
- Parse File storage > 80% quota
- Temporal workflow timeouts

### Potential Extensions

1. **Batch Status Endpoint**: `GET /v1/batch/{batch_id}/status`
2. **Batch Result Retrieval**: `GET /v1/batch/{batch_id}/results`
3. **Batch Templates**: Pre-defined formats with validation schemas
4. **Batch Scheduling**: Schedule batches for off-peak processing

---

## Appendix

### Configuration Constants

```python
# File: config/batch_memory.py

# Maximum number of memories per batch
MAX_BATCH_SIZE = 1000

# Maximum size of individual memory content (10MB)
MAX_MEMORY_SIZE = 10 * 1024 * 1024

# Chunk size for processing
BATCH_CHUNK_SIZE = 10

# Temporal workflow timeout
BATCH_WORKFLOW_TIMEOUT = timedelta(hours=2)

# Parse File storage settings
BATCH_FILE_EXPIRY_DAYS = 30  # Auto-delete old batch files
```

### Related Documentation

- **Parse File Storage**: [Parse Server Files Guide](https://docs.parseplatform.org/rest/guide/#files)
- **Temporal Durable Execution**: [Temporal Workflows](https://docs.temporal.io/workflows)
- **Existing Pattern**: `services/memory_management.py`
- **Architecture Document**: [BATCH_MEMORY_PAYLOAD_FIX_ARCHITECTURE.md](./BATCH_MEMORY_PAYLOAD_FIX_ARCHITECTURE.md)
- **Implementation Plan**: [BATCH_MEMORY_PAYLOAD_FIX_IMPLEMENTATION.md](./BATCH_MEMORY_PAYLOAD_FIX_IMPLEMENTATION.md)

---

**Document Version**: 1.0
**Last Review**: 2025-10-21
**Next Review**: 2025-11-21
