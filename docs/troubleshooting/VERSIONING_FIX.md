# Worker Versioning Fix for Child Workflows

## Problem
Child workflows (`ProcessBatchMemoryFromPostWorkflow` and `ProcessBatchMemoryWorkflow`) were running on **unversioned** workers instead of using the new `v0.2.0+batch-default.20251117` build ID. This caused them to use the old code path with `add_memory_quick` instead of the new batch processing path.

## Root Cause
When starting child workflows using `workflow.start_child_workflow()`, Temporal does **NOT** automatically inherit the parent workflow's build ID. You must explicitly specify `versioning_intent=VersioningIntent.VERSIONED` to ensure child workflows use the same build ID as the parent.

##  Changes Made

### 1. `document_processing.py`
- ✅ Added `VersioningIntent` import
- ✅ Added `versioning_intent=VersioningIntent.VERSIONED` to child workflow call (line 421)

```python
from temporalio.common import RetryPolicy, VersioningIntent

batch_workflow_handle = await workflow.start_child_workflow(
    ProcessBatchMemoryFromPostWorkflow.run,
    args=[{...}],
    id=f"batch-memory-doc-{upload_id}",
    task_queue="memory-processing",
    versioning_intent=VersioningIntent.VERSIONED  # ← NEW: Use versioned worker
)
```

### 2. `batch_memory.py`
- ✅ Added `VersioningIntent` import
- ✅ Added `versioning_intent=VersioningIntent.VERSIONED` to child workflow call (line 665)
- ✅ Added clarifying comments to API helper functions explaining auto-routing

```python
from temporalio.common import RetryPolicy, VersioningIntent

batch_workflow_handle = await workflow.start_child_workflow(
    ProcessBatchMemoryWorkflow.run,
    args=[workflow_data],
    id=unique_workflow_id,
    task_queue="memory-processing",
    versioning_intent=VersioningIntent.VERSIONED,  # ← NEW: Use versioned worker
    task_timeout=timedelta(minutes=30)
)
```

## How Worker Versioning Works

### For Child Workflows (Inside Workflows)
**❌ Without `versioning_intent`:**
- Child workflows get routed to **any available worker** on the task queue
- Could be old unversioned workers or new versioned workers
- **Result:** Inconsistent behavior, may use old code

**✅ With `versioning_intent=VersioningIntent.VERSIONED`:**
- Child workflows inherit the parent's build ID
- Guaranteed to run on the same worker version as the parent
- **Result:** Consistent behavior, always uses new code

### For API-Started Workflows (Outside Workflows)
Workflows started via `client.start_workflow()` from API routes automatically use the **default build ID** configured on the task queue. No changes needed if the task queue has versioning rules set up.

## Expected Behavior After Fix

### Before (❌ Broken)
```
DocumentProcessingWorkflow (v0.2.0+batch-default.20251117)
  └─> ProcessBatchMemoryFromPostWorkflow (unversioned) ❌
        └─> ProcessBatchMemoryWorkflow (unversioned) ❌
              └─> Uses add_memory_quick (OLD CODE) ❌
```

### After (✅ Fixed)
```
DocumentProcessingWorkflow (v0.2.0+batch-default.20251117)
  └─> ProcessBatchMemoryFromPostWorkflow (v0.2.0+batch-default.20251117) ✅
        └─> ProcessBatchMemoryWorkflow (v0.2.0+batch-default.20251117) ✅
              └─> Uses batch_add_memory_quick (NEW CODE) ✅
```

## Deployment Steps

### 1. Restart Workers
```bash
# Stop all workers
pkill -f start_all_workers.py
pkill -f start_temporal_worker.py
pkill -f start_document_worker.py

# Verify nothing is running
ps aux | grep -E "start_(all|temporal|document)_worker" | grep -v grep

# Start workers with new code
cd /Users/shawkatkabbara/Documents/GitHub/memory
nohup python3 start_all_workers.py > logs/start_all_workers.log 2> logs/start_all_workers.err &

# Verify workers are running
ps aux | grep start_all_workers.py | grep -v grep
```

### 2. Verify Worker Build ID
Check that workers registered with the new build ID:

```bash
# Check worker logs
tail -f logs/start_all_workers.log
# Should see: "🏗️  Worker build ID: v0.2.0+batch-default.20251117"
```

### 3. Test a New Workflow
Run your test again and check Temporal UI:

```bash
poetry run pytest tests/test_document_processing_v2.py::test_document_upload_v2_with_real_pdf_file_custom_schema -v -s
```

### 4. Verify in Temporal UI
In the Temporal Cloud UI, check:

1. **Parent workflow** (`DocumentProcessingWorkflow`):
   - SDK: Should show `unversioned:v0.2.0+batch-default.20251117`
   - ✅ This is correct

2. **Child workflow #1** (`ProcessBatchMemoryFromPostWorkflow`):
   - SDK: Should NOW show `unversioned:v0.2.0+batch-default.20251117` (not just `unversioned`)
   - ✅ Fixed by adding `versioning_intent`

3. **Child workflow #2** (`ProcessBatchMemoryWorkflow`):
   - SDK: Should NOW show `unversioned:v0.2.0+batch-default.20251117` (not just `unversioned`)
   - ✅ Fixed by adding `versioning_intent`

### 5. Check Logs for Batch Processing
Look for the new log messages in `logs/start_all_workers.log`:

```
✅ default-batch-processing patch active: using batch processing
```

**NOT** the old message:
```
⚠️ No deprecation patch detected for direct workflow: using individual processing
```

## Verification Checklist

- [ ] Workers restarted with new code
- [ ] Workers show build ID `v0.2.0+batch-default.20251117` in logs
- [ ] New test workflow created
- [ ] Parent workflow shows versioned build ID in Temporal UI
- [ ] Child workflow #1 shows versioned build ID (not just "unversioned")
- [ ] Child workflow #2 shows versioned build ID (not just "unversioned")
- [ ] Logs show "default-batch-processing patch active"
- [ ] Logs show "batch_add_memory_quick" activity calls
- [ ] Logs do NOT show "add_memory_quick" calls for new workflows

## Technical Details

### What is `VersioningIntent.VERSIONED`?
From Temporal docs:
- `VERSIONED`: Child workflow uses the **same build ID** as the parent workflow
- `DEFAULT`: Child workflow uses the **default build ID** from task queue assignment rules
- `INHERIT`: (deprecated) Child workflow inherits parent's settings

For our use case, we want `VERSIONED` to ensure all parts of the document processing pipeline use the same code version.

### Why This Matters
Without proper versioning:
1. Parent workflow runs new code (batch processing)
2. Child workflows run old code (individual processing)
3. Result: MongoDB connection pool exhaustion, slower processing, inconsistent behavior

With proper versioning:
1. Parent workflow runs new code
2. Child workflows run new code
3. Result: Fast batch processing, efficient connection usage, consistent behavior

## References
- [Temporal Python Versioning Docs](https://docs.temporal.io/develop/python/versioning)
- [Child Workflows Documentation](https://docs.temporal.io/develop/python/child-workflows)
- [Worker Versioning Guide](https://docs.temporal.io/workers#worker-versioning)

