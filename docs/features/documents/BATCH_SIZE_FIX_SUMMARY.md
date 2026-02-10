# BatchMemoryRequest Validation Error Fix

## Problem
The document processing workflow was failing with:
```
1 validation error for BatchMemoryRequest
memories
  List should have at most 50 items after validation, not 359
```

## Root Cause
When processing a 30-page document with complex content, the `generate_llm_optimized_memory_structures` activity generated 359 `AddMemoryRequest` objects. The code then tried to create a single `BatchMemoryRequest` with all 359 items, but `BatchMemoryRequest` has a Pydantic validation constraint of **maximum 50 items**.

## Solution

### 1. Removed BatchMemoryRequest Creation from `generate_llm_optimized_memory_structures`
**File:** `cloud_plugins/temporal/activities/document_activities.py`
**Lines:** ~1605-1625

**Before:**
```python
if memory_requests:
    from models.memory_models import BatchMemoryRequest
    batch_request = BatchMemoryRequest(
        external_user_id=metadata.user_id,
        organization_id=organization_id,
        namespace_id=namespace_id,
        memories=memory_requests,  # Could be 359 items!
        batch_size=min(20, len(memory_requests))
    )
    response["batch_request"] = batch_request.model_dump()
```

**After:**
```python
# Return memory requests as dicts - batching will be handled by the workflow
# BatchMemoryRequest has a max limit of 50 items, so we don't create it here
response: Dict[str, Any] = {
    "memory_requests": [req.model_dump() for req in memory_requests],
    "generation_summary": generated_summary,
    "domain": domain,
    "llm_optimized": True,
    "total_generated": len(memory_requests),
    "external_user_id": metadata.user_id,
    "organization_id": organization_id,
    "namespace_id": namespace_id
}
```

### 2. Added Chunking Logic to `create_hierarchical_memory_batch`
**File:** `cloud_plugins/temporal/activities/document_activities.py`
**Lines:** ~1361-1444

**Changes:**
- Added `MAX_BATCH_SIZE = 50` constant
- Split memories into chunks of 50 items
- Process each chunk separately with its own `BatchMemoryRequest`
- Track progress with activity heartbeats for each chunk
- Return aggregated results from all batches

**Key Code:**
```python
# BatchMemoryRequest has a max limit of 50 items, so split into chunks
MAX_BATCH_SIZE = 50
total_memories_created = 0
batch_results = []

for i in range(0, len(memories), MAX_BATCH_SIZE):
    chunk = memories[i:i + MAX_BATCH_SIZE]
    chunk_num = (i // MAX_BATCH_SIZE) + 1
    total_chunks = (len(memories) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
    
    activity.heartbeat(f"Processing batch {chunk_num}/{total_chunks} ({len(chunk)} items)")
    
    # Create batch request for this chunk
    batch_request = BatchMemoryRequest(
        external_user_id=user_id,
        organization_id=organization_id,
        namespace_id=namespace_id,
        memories=chunk,
        batch_size=min(batch_size, len(chunk))
    )

    # Process batch
    batch_result = await process_batch_with_temporal(...)
    batch_results.append(batch_result)
    total_memories_created += len(chunk)
```

## Benefits
1. ✅ **Respects Pydantic validation constraints** - Never exceeds 50 items per BatchMemoryRequest
2. ✅ **Handles large documents** - Can process any number of memories by chunking
3. ✅ **Progress tracking** - Activity heartbeats show progress for each chunk
4. ✅ **Complete results** - Returns aggregated results from all batches
5. ✅ **No data loss** - All 359 memories will be processed across 8 batches (7×50 + 1×9)

## Example Workflow
For a document with 359 memories:
- **Batch 1:** Memories 0-49 (50 items)
- **Batch 2:** Memories 50-99 (50 items)
- **Batch 3:** Memories 100-149 (50 items)
- **Batch 4:** Memories 150-199 (50 items)
- **Batch 5:** Memories 200-249 (50 items)
- **Batch 6:** Memories 250-299 (50 items)
- **Batch 7:** Memories 300-349 (50 items)
- **Batch 8:** Memories 350-358 (9 items)

**Total:** 359 memories processed successfully in 8 batches ✅

## Testing
To verify the fix works:
```bash
poetry run pytest tests/test_document_processing_v2.py::test_document_upload_v2_with_real_pdf_file -v -s
```

The test should now complete successfully without validation errors.

