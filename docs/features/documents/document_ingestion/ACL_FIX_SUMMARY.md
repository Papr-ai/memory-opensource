# ACL Fields Fix for Document Upload

## Problem
When documents were uploaded via `/v1/document/v2` and processed through Temporal workflows, the created memories and Neo4j nodes had empty ACL fields (`user_read_access`, `user_write_access`, etc.), making them inaccessible.

## Root Cause
The `create_memory_batch_for_pages` activity was creating memories with basic metadata dictionaries that didn't include ACL fields. These memories bypassed the normal ACL-setting logic in `handle_incoming_memory`.

## Solution

### 1. User ID Resolution (Already Working)
- API key → resolved to developer's internal user_id
- Bearer token → resolved from token to internal user_id
- End user ID → resolved to internal user_id
- The **resolved user_id should ALWAYS have access** to memories they create

### 2. ACL Merging Strategy
Instead of overwriting user-provided ACLs, we now **merge** them:
- If user provides ACLs in `MemoryMetadata`, preserve them
- Always add the resolved `user_id` to the ACL lists
- Use `merge_acl_lists()` helper: `list(set((existing or []) + (new or [])))`

This follows the same pattern as `handle_incoming_memory` in `services/memory_service.py`.

## Changes Made

### File: `services/user_utils.py`

#### Function: `patch_and_resolve_user_ids_and_acls` (lines 3086-3131)

**Problem:**
- Was **overwriting** `user_read_access` and `user_write_access` with only resolved external IDs
- Didn't ensure resolved `end_user_id` was always added to ACLs
- Lost any user-provided internal user IDs in ACL lists

**After:**
- **Merges** resolved external IDs with existing `user_read_access`/`user_write_access`
- **Always adds** resolved `end_user_id` to both read and write access lists
- Preserves user-provided internal user IDs
- Uses `list(set(...))` to deduplicate merged ACL lists

```python
# Initialize ACL lists if they don't exist
if not getattr(meta, 'user_read_access', None):
    meta.user_read_access = []
if not getattr(meta, 'user_write_access', None):
    meta.user_write_access = []

# CRITICAL: Always ensure the resolved end_user_id has access
if end_user_id and end_user_id not in meta.user_read_access:
    meta.user_read_access.append(end_user_id)
if end_user_id and end_user_id not in meta.user_write_access:
    meta.user_write_access.append(end_user_id)

# Resolve external IDs and MERGE with existing user ACLs (don't overwrite)
if getattr(meta, 'external_user_read_access', None):
    resolved_user_ids = [...]  # Resolve external IDs to internal
    meta.user_read_access = list(set(meta.user_read_access + resolved_user_ids))
```

This ensures:
1. Resolved user always has access
2. User-provided ACLs are preserved and merged
3. External user IDs are properly resolved and added
4. No duplicate user IDs in ACL lists

### File: `cloud_plugins/temporal/activities/document_activities.py`

#### Function: `create_memory_batch_for_pages` (lines 886-1007)

**Before:**
- Created plain metadata dictionaries without ACL fields
- Memories had no access control

**After:**
- Extracts any existing metadata/ACLs from page_data
- Merges resolved user_id with existing ACLs using `merge_acl_lists()`
- Creates proper `MemoryMetadata` objects with complete ACL fields
- Ensures the user who uploaded always has access, plus any ACLs they specified

```python
# Get existing ACL fields from user-provided metadata (if any)
existing_user_read = existing_metadata.get("user_read_access", [])
existing_user_write = existing_metadata.get("user_write_access", [])
# ... other ACL fields ...

# Merge resolved user_id with any existing ACLs
merged_user_read = merge_acl_lists(existing_user_read, [user_id])
merged_user_write = merge_acl_lists(existing_user_write, [user_id])

memory_metadata = MemoryMetadata(
    user_id=user_id,
    workspace_id=workspace_id,
    user_read_access=merged_user_read,
    user_write_access=merged_user_write,
    # ... other fields ...
)
```

### File: `routers/v1/document_routes_v2.py`

#### Function: `upload_document_v2` (lines 315-333)

**Added:**
- Ensures resolved user is added to ACLs when enriching metadata
- Merges with any existing user-provided ACLs (doesn't overwrite)

```python
# Ensure resolved user has access to the document
if not metadata.user_read_access:
    metadata.user_read_access = []
if not metadata.user_write_access:
    metadata.user_write_access = []

# Add resolved user to ACLs if not already present
if end_user_id_resolved and end_user_id_resolved not in metadata.user_read_access:
    metadata.user_read_access.append(end_user_id_resolved)
if end_user_id_resolved and end_user_id_resolved not in metadata.user_write_access:
    metadata.user_write_access.append(end_user_id_resolved)
```

## ACL Flow Through System

### Batch Memory Route (`/v1/memory/batch`)

1. **Batch Request Received**
   - User provides `BatchMemoryRequest` with memories
   - Each memory may have `MemoryMetadata` with ACLs already set
   - User may use `external_user_id` instead of internal `user_id`

2. **Authentication & User Resolution**
   - `get_user_from_token_optimized` resolves auth credentials
   - Calls `patch_and_resolve_user_ids_and_acls` to:
     - Resolve `external_user_id` → internal `user_id` (end_user_id)
     - Resolve external user IDs in ACL fields → internal user IDs
     - **Add resolved `end_user_id` to `user_read_access` and `user_write_access`**
     - **Merge** with existing user-provided ACLs (don't overwrite)

3. **Temporal Batch Processing** (if enabled)
   - Batch request passed to `process_batch_with_temporal`
   - All ACLs preserved in `BatchWorkflowData`
   - Memories processed through multi-stage pipeline
   - ACLs flow through to Parse and Neo4j storage

4. **Background Processing** (if Temporal disabled)
   - Batch request passed to `common_add_memory_batch_handler`
   - Each memory processed through `handle_incoming_memory`
   - ACLs may be further enriched based on context (postMessageId, pageId, workspace)
   - Final ACLs stored in Parse and Neo4j

### Document Upload Route (`/v1/document/v2`)

1. **Document Upload** (`/v1/document/v2`)
   - User provides optional `metadata: MemoryMetadata` with ACLs
   - Route resolves auth and adds resolved user to ACLs
   - Metadata passed to Temporal workflow

2. **Document Processing Workflow**
   - LLM generates memories from document pages
   - Each memory inherits ACLs from document metadata

3. **Memory Batch Creation** (`create_memory_batch_for_pages`)
   - Merges page-level metadata with document metadata
   - Ensures resolved user is in ACL lists
   - Creates proper `MemoryMetadata` with complete ACLs

4. **Storage** (Parse & Neo4j)
   - Parse: ACLs stored in Memory Parse objects (via `store_generic_memory_item`)
   - Neo4j: ACLs stored in node properties (via `store_llm_generated_graph`)
   
   From `memory_graph.py` lines 7823-7830:
   ```python
   common_metadata = {
       "user_id": metadata.get("user_id"),
       "user_read_access": metadata.get("user_read_access", []),
       "user_write_access": metadata.get("user_write_access", []),
       "workspace_read_access": metadata.get("workspace_read_access", []),
       # ... other ACL fields ...
   }
   ```

## Testing

### Manual Test
```bash
# Kill and restart workers to pick up changes
pkill -f start_temporal_worker.py
pkill -f start_document_worker.py

cd /Users/shawkatkabbara/Documents/GitHub/memory
python start_temporal_worker.py &
python start_document_worker.py &

# Run document upload test
poetry run pytest tests/test_document_processing_v2.py::test_document_upload_v2_with_real_pdf_file_custom_schema -v
```

### Expected Results
After fix, memories and Neo4j nodes should have:
- `user_read_access`: Contains resolved user_id + any user-provided user IDs
- `user_write_access`: Contains resolved user_id + any user-provided user IDs
- Other ACL fields: Preserved from user input or set to `[]`

### Verification Query
```cypher
MATCH (n:Memory)
WHERE n.upload_id = '<upload_id>'
RETURN 
  n.id,
  n.content[0..50] as preview,
  n.user_read_access,
  n.user_write_access,
  n.workspace_read_access,
  n.role_read_access
LIMIT 5
```

## Key Principles

1. **Always resolve to internal user_id** - Never use external IDs directly for ACLs
2. **Merge, don't overwrite** - Preserve user-provided ACLs while adding resolved user
3. **Default to private** - If no ACLs provided, set to user-only access
4. **Consistent behavior** - Document uploads should behave like `/v1/memory/batch`

## Related Code References

- `services/memory_service.py:handle_incoming_memory()` - Lines 247-270 (ACL logic pattern)
- `memory/memory_graph.py:store_llm_generated_graph()` - Lines 7823-7830 (Neo4j ACL storage)
- `services/memory_management.py:store_generic_memory_item()` - Lines 889-896 (Parse ACL storage)
- `routers/v1/memory_routes_v1.py:add_memory_batch_v1()` - Lines 820-910 (Batch route ACL handling)

