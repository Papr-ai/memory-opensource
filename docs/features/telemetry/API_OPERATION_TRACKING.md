# API Operation Tracking

## Overview

This system provides dual-level tracking of all API operations:

1. **Monthly Aggregates** (Parse Server `Interaction` class)
   - Used for rate limiting and quota enforcement
   - Accessible via Parse Dashboard
   - Monthly rollup by user, workspace, route, and method

2. **Detailed Logs** (MongoDB Time Series `api_operation_logs` collection)
   - Granular per-request logging
   - Optimized for analytics and time-based queries
   - 90-day retention (configurable)
   - Includes performance metrics, user context, and metadata

## Schema

### Interaction Class (Parse Server)

**Updated fields:**

```javascript
{
  objectId: String,
  createdAt: Date,
  updatedAt: Date,
  
  // Who
  user: Pointer<_User>,
  workspace: Pointer<WorkSpace>,
  organization: Pointer<Organization>,  // NEW
  subscription: Pointer<Subscription>,
  
  // What
  type: String,  // "mini" | "premium" | "api_operation"  (NEW enum value)
  route: String,  // NEW: API route (e.g., "v1/memory", "v1/user")
  method: String,  // NEW: HTTP method (GET, POST, PUT, DELETE)
  isMemoryOperation: Boolean,  // NEW: Counts against memory operation limits
  
  // When
  month: Number,  // 1-12
  year: Number,   // e.g., 2025
  
  // Count
  count: Number,  // Incremented on each API call
  
  // ACL
  ACL: Object
}
```

**Indexes:**

```javascript
// Existing index
{user: 1, workspace: 1, type: 1, month: 1, year: 1}

// NEW indexes for API operations
{organization: 1, type: 1, month: 1, year: 1}
{route: 1, method: 1, isMemoryOperation: 1, month: 1, year: 1}
```

### APIOperationLog Collection (MongoDB Time Series)

```javascript
{
  // Time (required for time series)
  timestamp: Date,  // When the operation occurred (UTC)
  
  // Who
  user_id: String,
  workspace_id: String,
  organization_id: String,
  developer_id: String,  // API key owner
  
  // What
  route: String,  // "v1/memory", "v1/memory/search", etc.
  method: String,  // "GET", "POST", "PUT", "DELETE"
  operation_type: String,  // "add_memory", "search_memory", etc.
  
  // Memory operation specific
  is_memory_operation: Boolean,
  memory_id: String,
  batch_size: Number,
  
  // Performance
  latency_ms: Number,
  status_code: Number,
  
  // Context
  api_key: String,  // Hashed
  client_type: String,
  ip_address: String,
  user_agent: String,
  
  // Metadata
  metadata: Object
}
```

**Time Series Configuration:**

```javascript
{
  timeField: "timestamp",
  metaField: "metadata",
  granularity: "seconds",
  expireAfterSeconds: 7776000  // 90 days
}
```

## Integration

### 1. Initialize Tracker

In your `app_factory.py`:

```python
from services.api_operation_tracker import get_api_operation_tracker

async def startup_event():
    tracker = get_api_operation_tracker()
    await tracker.initialize()
```

### 2. Track Operations in Routes

**Example: Add Memory Route**

```python
from services.api_operation_tracker import get_api_operation_tracker
import time

@router.post("/memory")
async def add_memory_v1(
    request: Request,
    memory_request: AddMemoryRequest,
    # ... other params
):
    start_time = time.time()
    
    try:
        # ... your route logic ...
        
        result = await memory_graph.add_memory(...)
        
        # Track the operation
        tracker = get_api_operation_tracker()
        await tracker.track_operation(
            user_id=end_user_id,
            workspace_id=workspace_id,
            route="v1/memory",
            method="POST",
            is_memory_operation=True,  # This counts against limits
            organization_id=organization_id,
            developer_id=developer_id,
            operation_type="add_memory",
            memory_id=result.data[0].memoryId if result.data else None,
            latency_ms=(time.time() - start_time) * 1000,
            status_code=200,
            api_key=api_key,
            client_type=request.headers.get('X-Client-Type'),
            ip_address=request.client.host,
            user_agent=request.headers.get('User-Agent')
        )
        
        return result
        
    except Exception as e:
        # Track error
        tracker = get_api_operation_tracker()
        await tracker.track_operation(
            user_id=end_user_id,
            workspace_id=workspace_id,
            route="v1/memory",
            method="POST",
            is_memory_operation=True,
            organization_id=organization_id,
            developer_id=developer_id,
            operation_type="add_memory",
            latency_ms=(time.time() - start_time) * 1000,
            status_code=500,
            api_key=api_key,
            client_type=request.headers.get('X-Client-Type')
        )
        raise
```

**Example: Search Route (Non-Memory Operation)**

```python
@router.post("/memory/search")
async def search_v1(...):
    start_time = time.time()
    
    # ... your search logic ...
    
    # Track search (counts as LLM interaction, not memory operation)
    tracker = get_api_operation_tracker()
    await tracker.track_operation(
        user_id=end_user_id,
        workspace_id=workspace_id,
        route="v1/memory/search",
        method="POST",
        is_memory_operation=False,  # Search doesn't count as memory operation
        organization_id=organization_id,
        operation_type="search_memory",
        latency_ms=(time.time() - start_time) * 1000,
        status_code=200,
        metadata={
            "query": search_request.query[:100],  # Truncate for logging
            "max_memories": max_memories,
            "enable_agentic_graph": search_request.enable_agentic_graph
        }
    )
```

### 3. Middleware Approach (Recommended)

For automatic tracking of all routes:

```python
# middleware/api_tracking.py
from fastapi import Request
from services.api_operation_tracker import get_api_operation_tracker
import time

async def track_api_operation_middleware(request: Request, call_next):
    """Middleware to automatically track all API operations"""
    start_time = time.time()
    
    # Extract route info
    route = request.url.path
    method = request.method
    
    # Determine if it's a memory operation
    is_memory_operation = (
        "memory" in route.lower() and
        method in ["POST", "PUT", "DELETE"] and
        "search" not in route.lower()
    )
    
    response = await call_next(request)
    
    # Track after response
    try:
        tracker = get_api_operation_tracker()
        
        # Get user context from request state (set by auth middleware)
        user_id = getattr(request.state, 'user_id', None)
        workspace_id = getattr(request.state, 'workspace_id', None)
        organization_id = getattr(request.state, 'organization_id', None)
        
        if user_id and workspace_id:
            await tracker.track_operation(
                user_id=user_id,
                workspace_id=workspace_id,
                route=route,
                method=method,
                is_memory_operation=is_memory_operation,
                organization_id=organization_id,
                latency_ms=(time.time() - start_time) * 1000,
                status_code=response.status_code,
                api_key=request.headers.get('X-API-Key'),
                client_type=request.headers.get('X-Client-Type'),
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get('User-Agent')
            )
    except Exception as e:
        logger.error(f"Failed to track API operation: {e}")
    
    return response

# In app_factory.py
app.middleware("http")(track_api_operation_middleware)
```

## Querying

### Parse Dashboard

View monthly aggregates:
1. Go to Parse Dashboard → Interaction class
2. Filter by:
   - `type = "api_operation"`
   - `route = "v1/memory"`
   - `isMemoryOperation = true`
   - `month = 10, year = 2025`

### MongoDB Queries

**Get operations for a user (last 24 hours):**

```javascript
db.api_operation_logs.find({
  user_id: "user_123",
  timestamp: {
    $gte: new Date(Date.now() - 24 * 60 * 60 * 1000)
  }
}).sort({timestamp: -1}).limit(100)
```

**Aggregate memory operations per day:**

```javascript
db.api_operation_logs.aggregate([
  {
    $match: {
      is_memory_operation: true,
      timestamp: {
        $gte: new Date("2025-10-01"),
        $lt: new Date("2025-11-01")
      }
    }
  },
  {
    $group: {
      _id: {
        $dateToString: {format: "%Y-%m-%d", date: "$timestamp"}
      },
      count: {$sum: 1},
      total_latency: {$sum: "$latency_ms"}
    }
  },
  {
    $project: {
      date: "$_id",
      count: 1,
      avg_latency: {$divide: ["$total_latency", "$count"]}
    }
  },
  {$sort: {date: 1}}
])
```

**Find slow operations:**

```javascript
db.api_operation_logs.find({
  latency_ms: {$gt: 1000}  // > 1 second
}).sort({latency_ms: -1}).limit(20)
```

**Operations by organization:**

```javascript
db.api_operation_logs.aggregate([
  {
    $match: {
      timestamp: {$gte: new Date("2025-10-01")}
    }
  },
  {
    $group: {
      _id: "$organization_id",
      total_operations: {$sum: 1},
      memory_operations: {
        $sum: {$cond: ["$is_memory_operation", 1, 0]}
      },
      avg_latency: {$avg: "$latency_ms"}
    }
  },
  {$sort: {total_operations: -1}}
])
```

## Rate Limiting

### Memory Operations

Only operations with `isMemoryOperation: true` count against limits:

```python
# In check_memory_limits()
from config.features import get_features

features = get_features()
tier_limits = features.get_tier_limits(effective_tier)
max_operations = tier_limits.get('max_memory_operations_per_month')

# Query monthly count from Interaction
interaction = db.Interaction.find_one({
    "user": user_id,
    "workspace": workspace_id,
    "type": "api_operation",
    "isMemoryOperation": True,
    "month": current_month,
    "year": current_year
})

if interaction and interaction['count'] >= max_operations:
    raise RateLimitError(...)
```

## Benefits

### Parse Server Tracking
✅ Integrated with ACLs and Parse Dashboard  
✅ Simple monthly aggregation for rate limiting  
✅ Backward compatible with existing Interaction records  
✅ Can filter by organization, user, route, method  

### Time Series Tracking
✅ Optimized storage and query performance  
✅ Automatic 90-day retention  
✅ Detailed per-request analytics  
✅ Performance monitoring  
✅ No impact on Parse Dashboard  

## Migration

### 1. Update Interaction Schema in Parse

Add fields to Interaction class in Parse Dashboard:
- `organization` (Pointer to Organization)
- `route` (String)
- `method` (String)
- `isMemoryOperation` (Boolean)

### 2. Create Indexes

```python
# Run this script once
python scripts/create_interaction_indexes.py
```

### 3. Initialize Time Series

The collection is created automatically on first use via `tracker.initialize()`.

## Testing

```python
# Test tracking
from services.api_operation_tracker import get_api_operation_tracker

tracker = get_api_operation_tracker()
await tracker.initialize()

await tracker.track_operation(
    user_id="test_user_123",
    workspace_id="test_workspace_456",
    route="v1/memory",
    method="POST",
    is_memory_operation=True,
    organization_id="org_test_789",
    operation_type="add_memory",
    latency_ms=123.45,
    status_code=200
)

# Verify in MongoDB
log = await tracker._time_series_collection.find_one({"user_id": "test_user_123"})
print(log)
```

## Performance

- **Time Series Insert**: ~1-2ms per operation
- **Parse Server Update**: ~5-10ms per operation (uses atomic increment)
- **Total Overhead**: ~10-15ms per API call (non-blocking)

All tracking is done **asynchronously** to not block the API response.

## Future Enhancements

1. **Real-time Dashboard**: Stream API operations to WebSocket dashboard
2. **Alerting**: Trigger alerts for anomalies (high latency, error spikes)
3. **Cost Tracking**: Calculate API costs per organization
4. **Quota Enforcement**: Auto-throttle based on tier limits
5. **Analytics Export**: Export to BigQuery, Snowflake for advanced analytics

---

**Status**: Ready for implementation  
**Last Updated**: 2025-10-03

