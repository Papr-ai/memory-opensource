# Integration Example: Adding API Operation Tracking to Existing Routes

## Example: Update add_memory_v1 Route

### Before (Current)

```python
@router.post("/memory")
async def add_memory_v1(
    request: Request,
    memory_request: AddMemoryRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    # ... other params
) -> AddMemoryResponse:
    try:
        # ... authentication ...
        
        # Add memory
        result = await common_add_memory_handler(...)
        
        response.status_code = result.code
        return result
        
    except Exception as e:
        logger.error(f"Error processing memory: {e}", exc_info=True)
        response.status_code = 500
        return AddMemoryResponse.failure(error=str(e), code=500)
```

### After (With Tracking)

```python
from services.api_operation_tracker import get_api_operation_tracker
import time

@router.post("/memory")
async def add_memory_v1(
    request: Request,
    memory_request: AddMemoryRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    # ... other params
) -> AddMemoryResponse:
    start_time = time.time()  # Track latency
    memory_id = None
    status_code = 500
    
    try:
        # ... authentication ...
        
        # Add memory
        result = await common_add_memory_handler(...)
        
        # Extract memory ID from result
        if result.data and len(result.data) > 0:
            memory_id = result.data[0].memoryId
        
        status_code = result.code
        response.status_code = result.code
        
        # Track operation (non-blocking)
        tracker = get_api_operation_tracker()
        await tracker.track_operation(
            user_id=end_user_id,
            workspace_id=workspace_id,
            route="v1/memory",
            method="POST",
            is_memory_operation=True,  # This counts against memory operation limits
            organization_id=organization_id,
            developer_id=developer_id,
            operation_type="add_memory",
            memory_id=memory_id,
            latency_ms=(time.time() - start_time) * 1000,
            status_code=status_code,
            api_key=api_key,
            client_type=request.headers.get('X-Client-Type'),
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get('User-Agent')
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing memory: {e}", exc_info=True)
        status_code = 500
        
        # Track error (non-blocking)
        tracker = get_api_operation_tracker()
        await tracker.track_operation(
            user_id=end_user_id if 'end_user_id' in locals() else None,
            workspace_id=workspace_id if 'workspace_id' in locals() else None,
            route="v1/memory",
            method="POST",
            is_memory_operation=True,
            organization_id=organization_id if 'organization_id' in locals() else None,
            operation_type="add_memory",
            latency_ms=(time.time() - start_time) * 1000,
            status_code=status_code,
            metadata={"error": str(e)[:200]}  # Truncate error for logging
        )
        
        response.status_code = 500
        return AddMemoryResponse.failure(error=str(e), code=500)
```

## Cleaner Approach: Helper Function

Create a helper to reduce boilerplate:

```python
# services/api_operation_tracker.py (add this method)

async def track_route_operation(
    route: str,
    method: str,
    request: Request,
    start_time: float,
    status_code: int,
    is_memory_operation: bool = False,
    operation_type: Optional[str] = None,
    memory_id: Optional[str] = None,
    batch_size: Optional[int] = None,
    error: Optional[str] = None
):
    """
    Helper to track route operations with minimal code
    
    Usage:
        await track_route_operation(
            route="v1/memory",
            method="POST",
            request=request,
            start_time=start_time,
            status_code=200,
            is_memory_operation=True,
            memory_id=result.memory_id
        )
    """
    from services.api_operation_tracker import get_api_operation_tracker
    
    # Extract from request state (set by auth middleware)
    user_id = getattr(request.state, 'user_id', None)
    end_user_id = getattr(request.state, 'end_user_id', None)
    workspace_id = getattr(request.state, 'workspace_id', None)
    organization_id = getattr(request.state, 'organization_id', None)
    developer_id = getattr(request.state, 'developer_id', None)
    
    if not user_id or not workspace_id:
        return  # Skip if no user context
    
    tracker = get_api_operation_tracker()
    
    metadata = {}
    if error:
        metadata['error'] = error[:200]  # Truncate
    
    await tracker.track_operation(
        user_id=end_user_id or user_id,
        workspace_id=workspace_id,
        route=route,
        method=method,
        is_memory_operation=is_memory_operation,
        organization_id=organization_id,
        developer_id=developer_id,
        operation_type=operation_type or tracker._infer_operation_type(route, method),
        memory_id=memory_id,
        batch_size=batch_size,
        latency_ms=(time.time() - start_time) * 1000,
        status_code=status_code,
        api_key=request.headers.get('X-API-Key'),
        client_type=request.headers.get('X-Client-Type'),
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get('User-Agent'),
        metadata=metadata if metadata else None
    )
```

Then use it in routes:

```python
@router.post("/memory")
async def add_memory_v1(...):
    start_time = time.time()
    
    try:
        # ... your logic ...
        result = await common_add_memory_handler(...)
        
        # Track (one line!)
        await track_route_operation(
            route="v1/memory",
            method="POST",
            request=request,
            start_time=start_time,
            status_code=result.code,
            is_memory_operation=True,
            memory_id=result.data[0].memoryId if result.data else None
        )
        
        return result
        
    except Exception as e:
        # Track error (one line!)
        await track_route_operation(
            route="v1/memory",
            method="POST",
            request=request,
            start_time=start_time,
            status_code=500,
            is_memory_operation=True,
            error=str(e)
        )
        raise
```

## Best Practice: Use Request State

Update your auth middleware to set these on `request.state`:

```python
# In get_user_from_token_optimized()
# After authentication succeeds:

request.state.user_id = user_id
request.state.end_user_id = end_user_id
request.state.workspace_id = workspace_id
request.state.organization_id = organization_id
request.state.developer_id = developer_id
```

Then tracking becomes even simpler!

## Routes to Update

### High Priority (Memory Operations - Count Against Limits)

1. ✅ `POST /v1/memory` - add_memory_v1
2. ✅ `POST /v1/memory/batch` - add_memory_batch_v1
3. ✅ `PUT /v1/memory/{id}` - update_memory_v1
4. ✅ `DELETE /v1/memory/{id}` - delete_memory_v1
5. ✅ `DELETE /v1/memory/all` - delete_all_memories_v1

### Medium Priority (Non-Memory Operations)

6. ⬜ `GET /v1/memory/{id}` - get_memory_v1
7. ⬜ `POST /v1/memory/search` - search_v1 (counts as LLM interaction)
8. ⬜ `POST /v1/user` - create_user
9. ⬜ `GET /v1/user` - get_user

## Full Integration Checklist

- [ ] Run setup script: `poetry run python scripts/setup_api_operation_tracking.py`
- [ ] Add tracker initialization to `app_factory.py`
- [ ] Choose approach: helper function or middleware
- [ ] Update request state in auth middleware
- [ ] Add tracking to memory operation routes
- [ ] Test with sample requests
- [ ] Verify in Parse Dashboard (Interaction class)
- [ ] Verify in MongoDB (api_operation_logs collection)
- [ ] Monitor performance impact (should be <15ms)

## Testing

```bash
# 1. Add a memory
curl -X POST http://localhost:8000/v1/memory \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content": "Test memory"}'

# 2. Check Parse Dashboard
# Go to Interaction class, filter by type="api_operation"

# 3. Check MongoDB
mongo
> use your_database
> db.api_operation_logs.find({route: "v1/memory"}).sort({timestamp: -1}).limit(1)

# Expected output:
{
  "timestamp": ISODate("2025-10-03T12:00:00Z"),
  "user_id": "user_123",
  "route": "v1/memory",
  "method": "POST",
  "is_memory_operation": true,
  "latency_ms": 123.45,
  "status_code": 200
}

# 4. Check monthly aggregate
> db.Interaction.findOne({
    type: "api_operation",
    route: "v1/memory",
    method: "POST"
  })

# Expected output:
{
  "_id": "...",
  "type": "api_operation",
  "route": "v1/memory",
  "method": "POST",
  "isMemoryOperation": true,
  "count": 42,  // Incremented on each operation
  "month": 10,
  "year": 2025
}
```

---

This approach gives you:
- ✅ **Dual tracking**: Parse Server aggregates + Time Series logs
- ✅ **Non-blocking**: All async background tasks
- ✅ **Minimal overhead**: ~10-15ms per request
- ✅ **Rich analytics**: Performance, errors, usage patterns
- ✅ **Rate limiting**: Memory operation counts for limits

