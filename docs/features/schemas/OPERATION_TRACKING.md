# Operation-Specific Interaction Tracking

This document explains how to use the new operation-specific interaction tracking system to accurately charge users based on the computational complexity of each API operation.

## Overview

The interaction tracking system now supports variable costs per operation. Instead of charging a flat 1 mini interaction for every API call, we now charge based on the actual computational work performed:

- **Simple operations** (1 mini interaction): Basic CRUD operations, no LLM usage
- **Complex operations** (4+ mini interactions): Multiple LLM calls, embeddings, graph processing
- **Variable operations** (1-3 mini interactions): Cost depends on features enabled (e.g., search with agentic graph)
- **Zero-cost operations** (0 mini interactions): Status checks, user feedback collection, document uploads (charged separately)

## Architecture

### Files

1. **`models/operation_types.py`**: Defines all operation types and their costs
2. **`config/cloud.yaml`**: Configuration for operation costs per tier
3. **`services/user_utils.py`**: Updated `check_interaction_limits_fast()` method

### Operation Cost Calculation

The `get_operation_cost()` function calculates the cost for each operation:

```python
from models.operation_types import MemoryOperationType, get_operation_cost

# Simple operation
cost = get_operation_cost(MemoryOperationType.UPDATE_MEMORY)  # Returns 1

# Batch operation
cost = get_operation_cost(
    MemoryOperationType.ADD_MEMORY_BATCH,
    batch_size=10  # Returns 40 (4 per memory)
)

# Variable cost operation
cost = get_operation_cost(
    MemoryOperationType.SEARCH_MEMORY,
    enable_agentic_graph=True,
    enable_rank_results=True  # Returns 3 (1 base + 1 agentic + 1 ranking)
)
```

## Operation Costs Reference

### Memory Operations
- `add_memory_v1`: **4 mini** (goal prediction + search+rank + graph building)
- `add_memory_batch_v1`: **4 × batch_size mini**
- `update_memory_v1`: **1 mini**
- `delete_memory_v1`: **1 mini**
- `delete_all_memories_v1`: **1 mini**
- `get_memory_v1`: **1 mini**
- `search_v1`: **1-3 mini** (base + optional agentic + optional ranking)

### Document Operations
- `upload_document`: **0 mini** (charged via document ingestion fees)
- `get_document_status`: **0 mini** (status check)
- `cancel_document_processing`: **0 mini**

### Feedback Operations
- `submit_feedback_v1`: **0 mini** (user feedback collection)
- `submit_batch_feedback_v1`: **0 mini**
- `get_feedback_by_id_v1`: **1 mini**

### GraphQL Operations
- `graphql_proxy`: **1 mini** (simple proxy)
- `graphql_playground`: **0 mini** (static HTML)

### Message Operations
- `store_message`: **1 mini**
- `get_session_history`: **1 mini**
- `get_session_status`: **1 mini**
- `process_session_messages`: **1 mini**

### Schema Operations
- `create_user_schema_v1`: **1 mini**
- `list_user_schemas_v1`: **1 mini**
- `get_user_schema_v1`: **1 mini**
- `update_user_schema_v1`: **1 mini**
- `delete_user_schema_v1`: **1 mini**

### Sync Operations
- `get_sync_tiers`: **3 mini** (embeddings + ranking)
- `get_sync_delta`: **1 mini**

## Usage in Route Handlers

### Basic Example (Memory Routes)

```python
from models.operation_types import MemoryOperationType

@router.post("")
async def add_memory_v1(
    request: Request,
    memory_request: AddMemoryRequest,
    # ... other params
) -> AddMemoryResponse:
    # Check interaction limits with operation type
    limit_check = await user.check_interaction_limits_fast(
        interaction_type='mini',
        memory_graph=memory_graph,
        operation=MemoryOperationType.ADD_MEMORY  # 4 mini interactions
    )
    
    if limit_check:
        response_dict, status_code, is_error = limit_check
        if is_error:
            response.status_code = status_code
            return AddMemoryResponse.failure(
                error=response_dict.get('error'),
                code=status_code,
                message=response_dict.get('message')
            )
    
    # Proceed with operation...
```

### Batch Operations Example

```python
@router.post("/batch")
async def add_memory_batch_v1(
    request: Request,
    batch_request: BatchMemoryRequest,
    # ... other params
) -> BatchMemoryResponse:
    batch_size = len(batch_request.memories)
    
    # Check limits with batch size (4 mini per memory)
    limit_check = await user.check_interaction_limits_fast(
        interaction_type='mini',
        memory_graph=memory_graph,
        operation=MemoryOperationType.ADD_MEMORY_BATCH,
        batch_size=batch_size  # Cost = 4 × batch_size
    )
    
    if limit_check:
        response_dict, status_code, is_error = limit_check
        if is_error:
            response.status_code = status_code
            return BatchMemoryResponse.failure(
                error=response_dict.get('error'),
                code=status_code
            )
    
    # Proceed with batch operation...
```

### Variable Cost Example (Search)

```python
@router.post("/search")
async def search_v1(
    request: Request,
    search_request: SearchRequest,
    # ... other params
) -> SearchResponse:
    # Extract feature flags from request
    enable_agentic = search_request.enable_agentic_graph
    enable_ranking = search_request.rank_results
    
    # Check limits with variable cost (1-3 mini)
    limit_check = await user.check_interaction_limits_fast(
        interaction_type='mini',
        memory_graph=memory_graph,
        operation=MemoryOperationType.SEARCH_MEMORY,
        enable_agentic_graph=enable_agentic,
        enable_rank_results=enable_ranking
    )
    
    if limit_check:
        response_dict, status_code, is_error = limit_check
        if is_error:
            response.status_code = status_code
            return SearchResponse.failure(
                error=response_dict.get('error'),
                code=status_code
            )
    
    # Proceed with search...
```

### Zero-Cost Operations (Document Upload)

```python
@router.post("")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    # ... other params
) -> DocumentUploadResponse:
    # Check limits - will return None immediately for zero-cost ops
    limit_check = await user.check_interaction_limits_fast(
        interaction_type='mini',
        memory_graph=memory_graph,
        operation=MemoryOperationType.UPLOAD_DOCUMENT  # 0 mini (charged separately)
    )
    
    # limit_check will be None for zero-cost operations
    
    # Proceed with document upload...
    # Note: Document ingestion fees are charged separately based on page count
```

## Migration Guide

### For Existing Route Handlers

1. **Import the operation type enum:**
   ```python
   from models.operation_types import MemoryOperationType
   ```

2. **Add the operation parameter to limit checks:**
   ```python
   # Before
   limit_check = await user.check_interaction_limits_fast(
       interaction_type='mini',
       memory_graph=memory_graph
   )
   
   # After
   limit_check = await user.check_interaction_limits_fast(
       interaction_type='mini',
       memory_graph=memory_graph,
       operation=MemoryOperationType.YOUR_OPERATION_NAME
   )
   ```

3. **For batch operations, add batch_size:**
   ```python
   limit_check = await user.check_interaction_limits_fast(
       interaction_type='mini',
       memory_graph=memory_graph,
       operation=MemoryOperationType.ADD_MEMORY_BATCH,
       batch_size=len(batch_request.memories)
   )
   ```

4. **For search operations, add feature flags:**
   ```python
   limit_check = await user.check_interaction_limits_fast(
       interaction_type='mini',
       memory_graph=memory_graph,
       operation=MemoryOperationType.SEARCH_MEMORY,
       enable_agentic_graph=search_request.enable_agentic_graph,
       enable_rank_results=search_request.rank_results
   )
   ```

## Document Ingestion Fees

Document operations (`upload_document`, `get_document_status`, `cancel_document_processing`) have **0 mini interaction cost** because they are charged separately via document ingestion fees:

- **Basic documents** (text/images): $1.50 per 100 pages
- **Advanced documents** (financial/tables): $2.75 per 100 pages
- **Complex documents** (scanned/healthcare): $5.00 per 100 pages

These fees are handled separately through Stripe metering for document processing.

## Logging and Monitoring

The updated system provides detailed logging:

```
Operation add_memory_v1 cost: 4 mini interactions
Fast check completed in 145.32ms (success)
MongoDB interaction count updated to: 104 (incremented by 4)
```

In case of limit exceeded:

```
Fast check completed in 158.21ms (limit exceeded)
Operation cost: 4 mini interactions, current_count: 1004, limit: 1000
```

## Error Responses

When a limit is exceeded, the error response includes operation details:

```json
{
  "error": "Interaction limit reached...",
  "message": "You've reached your monthly limit...",
  "current_count": 1004,
  "limit": 1000,
  "operation_cost": 4,
  "operation": "add_memory_v1",
  "tier": "developer",
  "is_trial": false
}
```

## Testing

When testing with operation-specific costs:

```python
# Test that batch operations charge correctly
batch_size = 5
limit_check = await user.check_interaction_limits_fast(
    interaction_type='mini',
    memory_graph=memory_graph,
    operation=MemoryOperationType.ADD_MEMORY_BATCH,
    batch_size=batch_size
)
# Should charge 4 × 5 = 20 mini interactions
```

## Benefits

1. **Fair pricing**: Users only pay for the computational work performed
2. **Transparent costs**: Clear mapping of operation → mini interactions
3. **Flexible tiers**: Different limits per tier configured in cloud.yaml
4. **Better tracking**: Detailed logs show exact costs per operation
5. **Zero-cost ops**: Status checks and feedback don't consume quota
6. **Batch accuracy**: Proper charging for batch operations based on size

## Operation Type Tracking

### Interaction Schema with operation_type

The `Interaction` class now includes an `operation_type` field that tracks **which specific operation** consumed the interactions:

```json
{
  "workspace": {"__type": "Pointer", "className": "WorkSpace", "objectId": "4YVBwQbdfP"},
  "user": {"__type": "Pointer", "className": "_User", "objectId": "jtKplF3Gft"},
  "type": "mini",
  "month": 11,
  "year": 2025,
  "count": 4,
  "operation_type": "add_memory_v1",  // Which operation consumed these interactions
  "subscription": {"__type": "Pointer", "className": "Subscription", "objectId": "ASdvx8zSEB"},
  "createdAt": "2025-11-10T12:34:56.789Z",
  "updatedAt": "2025-11-10T12:34:56.789Z",
  "objectId": "0cX2QqG2HE"
}
```

### How It Works

1. **Capture Operation**: When `check_interaction_limits_fast()` is called, it receives the `operation` parameter
2. **Extract Operation Name**: The operation enum value is extracted (e.g., `MemoryOperationType.ADD_MEMORY.value` → `"add_memory_v1"`)
3. **Store in Interaction**: The `operation_type` is stored in both MongoDB and Parse Server Interaction records
4. **Track Over Time**: Each interaction increment updates the `operation_type` to the latest operation

### Developer Dashboard Usage

This data enables developers to see **exactly** which operations are consuming their interaction quota:

```
Mini Interactions This Month: 247 / 1000

Breakdown by Operation:
- add_memory_v1:       120 interactions (30 calls × 4 interactions each)
- search_v1:            75 interactions (50 calls × ~1.5 avg)
- get_sync_tiers:       36 interactions (12 calls × 3 each)
- update_memory_v1:     10 interactions (10 calls × 1 each)
- get_memory_v1:         6 interactions (6 calls × 1 each)
```

### Query Example

To get usage breakdown by operation:

```javascript
// Parse Server query
Parse.Cloud.run("getInteractionsByOperation", {
  workspaceId: "4YVBwQbdfP",
  month: 11,
  year: 2025
}).then(results => {
  // Returns aggregated counts by operation_type
  console.log(results);
  // {
  //   "add_memory_v1": 120,
  //   "search_v1": 75,
  //   "get_sync_tiers": 36,
  //   ...
  // }
});
```

### Notes

- **Last Write Wins**: If an interaction record is updated multiple times, `operation_type` reflects the most recent operation
- **Aggregation Needed**: For per-operation totals, aggregate across all interaction records for the month/year
- **Backwards Compatible**: Existing interactions without `operation_type` will show as `"unknown"` or can be filtered out

