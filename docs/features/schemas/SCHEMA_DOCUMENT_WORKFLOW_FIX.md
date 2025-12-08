# Schema ID & Document Workflow Architecture Fix

## Issues Identified

### 1. ✅ `schema_id` Not Passed to Document Workflow
- **Problem**: `schema_id` was extracted from metadata in `document_routes_v2.py` but not passed to `DocumentProcessingWorkflow`
- **Fix**: 
  - Extract `schema_id` from metadata JSON
  - Pass as parameter to `DocumentProcessingWorkflow.run()`
  - Store in workflow state
  - Pass to batch memory processing activities

### 2. ✅ `process_batch_memories_from_parse_reference` Not Doing Full Indexing
- **Problem**: Called `common_add_memory_batch_handler` with `skip_background_processing=True`, which only did **quick add**, not full LLM indexing, relationship building, or metrics
- **Fix**: 
  - Changed `skip_background_processing=True` to ensure synchronous processing
  - Injected `schema_id` into memory metadata before processing
  - Full pipeline now runs: quick add → LLM schema generation → relationships → metrics

### 3. ✅ `schema_id` Should Be Direct Parameter
- **Problem**: Extracted from batch memory metadata instead of being passed explicitly
- **Fix**: Added `schema_id` parameter to `process_batch_memories_from_parse_reference`

### 4. ✅ Duplicate `process_batch_with_temporal` Calls
- **Problem**: Called in both `add_memory_batch_v1` and `common_add_memory_batch_handler`, triggering workflow twice
- **Fix**: Removed from `common_add_memory_batch_handler`, kept only in route handler

### 5. ✅ Document Workflow Architecture (Option A - Child Workflow)
- **Problem**: Unclear separation between document processing and batch memory indexing
- **Fix**: 
  - `DocumentProcessingWorkflow` now:
    1. Processes document with provider
    2. Extracts/generates memory requests
    3. Stores memories in Parse Post (with `schema_id` in metadata)
    4. **Starts `ProcessBatchMemoryFromPostWorkflow` as child workflow**
    5. Waits for child workflow to complete
    6. Full multi-stage indexing pipeline runs in child workflow
  - **Benefits of Child Workflow Approach**:
    - Clean separation of concerns
    - Update batch memory workflow in one place
    - Better observability in Temporal UI
    - Child workflow can be reused from other contexts

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Document Upload (document_routes_v2.py)                         │
│ - Extract schema_id from metadata                               │
│ - Start DocumentProcessingWorkflow with schema_id               │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ DocumentProcessingWorkflow.run()                                 │
│ ┌───────────────────────────────────────────────────────────┐  │
│ │ Step 1: Download & Validate File                          │  │
│ └───────────────────────────────────────────────────────────┘  │
│ ┌───────────────────────────────────────────────────────────┐  │
│ │ Step 2: Process with Provider (Reducto/TensorLake)        │  │
│ └───────────────────────────────────────────────────────────┘  │
│ ┌───────────────────────────────────────────────────────────┐  │
│ │ Step 3: Extract Structured Content                        │  │
│ └───────────────────────────────────────────────────────────┘  │
│ ┌───────────────────────────────────────────────────────────┐  │
│ │ Step 4: LLM Memory Generation                             │  │
│ └───────────────────────────────────────────────────────────┘  │
│ ┌───────────────────────────────────────────────────────────┐  │
│ │ Step 5: Store Memories in Parse Post                      │  │
│ │ Activity: store_batch_memories_in_parse_for_processing    │  │
│ │ - Injects schema_id into each memory's customMetadata     │  │
│ │ - Reuses document Post to avoid duplicates                │  │
│ └───────────────────────────────────────────────────────────┘  │
│ ┌───────────────────────────────────────────────────────────┐  │
│ │ Step 6: Start Child Workflow (ProcessBatchMemoryFromPost)│  │
│ │ workflow.start_child_workflow()                           │  │
│ │   ↓                                                       │  │
│ │   ProcessBatchMemoryFromPostWorkflow.run()               │  │
│ │   ├─ Activity: process_batch_memories_from_parse_ref     │  │
│ │   │   ├─ Fetch memories from Parse Post                  │  │
│ │   │   ├─ Build BatchMemoryRequest with schema_ids        │  │
│ │   │   └─ Call common_add_memory_batch_handler            │  │
│ │   │       (skip_background_processing=True, synchronous) │  │
│ │   └─ For each memory (parallel, max 50):                 │  │
│ │       ├─ Stage 1: add_memory_quick                       │  │
│ │       ├─ Stage 2: idx_generate_graph_schema              │  │
│ │       │           (with schema_id enforcement)           │  │
│ │       ├─ Stage 3: update_relationships                   │  │
│ │       └─ Stage 4: idx_update_metrics                     │  │
│ └───────────────────────────────────────────────────────────┘  │
│ ┌───────────────────────────────────────────────────────────┐  │
│ │ Step 7: Webhook Notification (if configured)              │  │
│ └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Schema ID Flow

```
Document Upload Metadata
  └─> DocumentProcessingWorkflow (schema_id param)
      └─> store_batch_memories_in_parse_for_processing (schema_id param)
          └─> Inject schema_id into memory.metadata.customMetadata.schema_id
              └─> process_batch_memories_from_parse_reference (schema_id param)
                  └─> Build AddMemoryRequest with schema_ids=[schema_id]
                      └─> common_add_memory_batch_handler
                          └─> common_add_memory_handler (per memory)
                              └─> handle_incoming_memory
                                  └─> memory_graph.add_memory_item_async
                                      └─> _index_memories_and_process (extracts schema_ids from metadata)
                                          └─> generate_memory_graph_schema_async (schema_ids param)
                                              └─> LLM enforces custom schema
```

## Key Changes

### routers/v1/document_routes_v2.py
- Extract `schema_id` from metadata JSON
- Pass `schema_id` to `DocumentProcessingWorkflow.run()`

### cloud_plugins/temporal/workflows/document_processing.py
- Accept `schema_id` parameter in `run()`
- Pass `schema_id` to `store_batch_memories_in_parse_for_processing`
- **Start `ProcessBatchMemoryFromPostWorkflow` as child workflow** (Option A)
- Pass `schema_id` in child workflow args
- Wait for child workflow to complete before proceeding
- Removed `link_batch_memories_to_post` call (no longer needed)

### cloud_plugins/temporal/activities/document_activities.py
- **NEW**: `store_batch_memories_in_parse_for_processing` activity
  - Injects `schema_id` into memory metadata
  - Reuses existing document Post

### cloud_plugins/temporal/workflows/batch_memory.py
- Updated `ProcessBatchMemoryFromPostWorkflow.run()`:
  - Accept `schema_id` in `ref_data` dict parameter
  - Pass `schema_id` to `process_batch_memories_from_parse_reference` activity
  - Log schema_id for debugging

### cloud_plugins/temporal/activities/memory_activities.py
- Updated `process_batch_memories_from_parse_reference`:
  - Accept `schema_id` parameter (instead of extracting from memory)
  - Inject `schema_id` into memory metadata before processing
  - Build `BatchMemoryRequest` with `schema_ids=[schema_id]`
  - Call `common_add_memory_batch_handler` with `skip_background_processing=True`
  - This activity is called from `ProcessBatchMemoryFromPostWorkflow`

### routes/memory_routes.py
- Removed duplicate `should_use_temporal` check in `common_add_memory_batch_handler`
- Added comment explaining temporal routing happens at route level

### start_document_worker.py
- Register `store_batch_memories_in_parse_for_processing` activity
- **No need to register memory activities** - child workflow runs on `memory-processing` queue

## Neo4j Query for Verification

```cypher
# View SecurityPolicy nodes and 3 levels of relationships
MATCH path = (n:SecurityPolicy)-[*1..3]-(related)
RETURN path
LIMIT 50

# Verify nodes created with custom schema
MATCH (n)
WHERE n.schema_id = 'Dh6EivRmo8'  # Security Behaviors & Risk schema
RETURN DISTINCT labels(n) as NodeTypes, count(*) as Count
ORDER BY Count DESC

# Check all node types created for a specific document
MATCH (n)
WHERE n.upload_id = '<your_upload_id>'
RETURN DISTINCT labels(n) as NodeTypes, count(*) as Count
ORDER BY Count DESC
```

## Testing

1. **Upload document with custom schema**:
```bash
curl -X POST http://localhost:8000/v1/document/v2 \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@document.pdf" \
  -F "metadata={\"schema_id\":\"Dh6EivRmo8\",\"source\":\"test\"}" \
  -F "preferred_provider=reducto" \
  -F "hierarchical_enabled=true"
```

2. **Verify in Temporal UI**:
   - Check `DocumentProcessingWorkflow` execution
   - Verify `store_batch_memories_in_parse_for_processing` activity
   - Verify `process_batch_memories_from_parse_reference` activity
   - Check for sub-activities: `add_memory_quick`, `idx_generate_graph_schema`, `update_relationships`, `idx_update_metrics`

3. **Verify in Neo4j**:
```cypher
MATCH (n) WHERE n.upload_id = '<upload_id>'
RETURN DISTINCT labels(n), n.schema_id, count(*)
```

Expected: Custom schema node types (e.g., `SecurityBehavior`, `Control`, `RiskIndicator`) instead of system defaults (`Memory`, `Goal`, `UseCase`)

## Remaining TODOs

- [ ] Test with real document upload
- [ ] Verify Neo4j nodes match custom schema
- [ ] Verify no duplicate Post records
- [ ] Test with multiple schemas
- [ ] Performance testing for large documents (1000+ pages)

