# paprDB Node Constraints Architecture

## 🎯 Overview

This document describes the architecture for implementing **node constraints** with paprDB, supporting:
- ✅ Local-first constraint application on device
- ✅ Background sync to cloud
- ✅ Delta sync from cloud to device
- ✅ GraphQL queries with custom ontology
- ✅ Constraint definitions syncing

---

## 📐 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEVICE (Local-First)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         User Action: Add/Index Memory                    │ │
│  │  - User adds memory via SDK                              │ │
│  │  - Or: Background indexing from local files             │ │
│  └──────────────────┬───────────────────────────────────────┘ │
│                     │                                         │
│  ┌──────────────────▼───────────────────────────────────────┐ │
│  │      Step 1: Apply Constraints Locally                   │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │  paprDB.create_node(node_type, properties)         │  │ │
│  │  │  1. Fetch constraints from local SQLite            │  │ │
│  │  │  2. Check 'when' conditions                        │  │ │
│  │  │  3. Apply 'force' (override values)                │  │ │
│  │  │  4. Search for existing (if needed)                │  │ │
│  │  │  5. Apply 'merge' (if existing found)              │  │ │
│  │  │  6. Store in local SQLite                          │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  │                                                          │ │
│  │  Result: Node stored locally with constraints applied   │ │
│  └──────────────────┬───────────────────────────────────────┘ │
│                     │                                         │
│  ┌──────────────────▼───────────────────────────────────────┐ │
│  │      Step 2: Queue for Background Sync                  │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │  sync_queue.add({                                 │  │ │
│  │  │    node_id: "node_123",                           │  │ │
│  │  │    node_type: "Project",                          │  │ │
│  │  │    properties: {...},                             │  │ │
│  │  │    constraints_applied: true,                     │  │ │
│  │  │    constraint_version: 1,                         │  │ │
│  │  │    sync_status: "pending"                          │  │ │
│  │  │  })                                               │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  └──────────────────┬───────────────────────────────────────┘ │
│                     │                                         │
│  ┌──────────────────▼───────────────────────────────────────┐ │
│  │      Step 3: Background Sync Worker                    │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │  async def sync_worker():                         │  │ │
│  │  │    while True:                                     │  │ │
│  │  │      items = sync_queue.get_pending()             │  │ │
│  │  │      for item in items:                            │  │ │
│  │  │        await sync_to_cloud(item)                  │  │ │
│  │  │        sync_queue.mark_synced(item.id)            │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  └──────────────────┬───────────────────────────────────────┘ │
│                     │                                         │
│                     │ POST /v1/memories (with constraints)    │
│                     │                                         │
└─────────────────────┼─────────────────────────────────────────┘
                      │
                      │
┌─────────────────────▼─────────────────────────────────────────┐
│                    CLOUD (Source of Truth)                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │      Step 4: Cloud Receives & Validates                 │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │  POST /v1/memories                                 │  │ │
│  │  │  - Receives node with constraints_applied flag    │  │ │
│  │  │  - Validates constraints match cloud definitions   │  │ │
│  │  │  - Re-applies constraints (if version mismatch)    │  │ │
│  │  │  - Stores in Neo4j + Qdrant                        │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  └──────────────────┬───────────────────────────────────────┘ │
│                     │                                         │
│  ┌──────────────────▼───────────────────────────────────────┐ │
│  │      Step 5: Constraint Definitions Sync                 │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │  GET /v1/sync/constraints?since=<timestamp>       │  │ │
│  │  │  - Returns constraint definitions                  │  │ │
│  │  │  - Device updates local constraint cache           │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  └──────────────────┬───────────────────────────────────────┘ │
│                     │                                         │
│  ┌──────────────────▼───────────────────────────────────────┐ │
│  │      Step 6: Delta Sync (Cloud → Device)               │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │  GET /v1/sync/delta?cursor=<cursor>               │  │ │
│  │  │  - Returns nodes updated since cursor              │  │ │
│  │  │  - Includes constraint changes                     │  │ │
│  │  │  - Device updates local SQLite                     │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  └──────────────────┬───────────────────────────────────────┘ │
│                     │                                         │
└─────────────────────┼─────────────────────────────────────────┘
                      │
                      │ Delta response
                      │
┌─────────────────────▼─────────────────────────────────────────┐
│                    DEVICE (Update Local)                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │      Step 7: Apply Delta Updates                        │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │  paprDB.apply_delta(delta_items)                   │  │ │
│  │  │  - Updates nodes in local SQLite                   │  │ │
│  │  │  - Applies new constraints if needed               │  │ │
│  │  │  - Updates constraint definitions                  │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │      Step 8: GraphQL Query (Local)                      │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │  paprDB.graphql_query("""                          │  │ │
│  │  │    query {                                         │  │ │
│  │  │      project(id: "123") {                           │  │ │
│  │  │        name                                        │  │ │
│  │  │        tasks { title }                             │  │ │
│  │  │      }                                             │  │ │
│  │  │    }                                               │  │ │
│  │  │  """)                                              │  │ │
│  │  │  - Uses custom ontology from local schema          │  │ │
│  │  │  - Returns results from local SQLite               │  │ │
│  │  │  - Works OFFLINE                                   │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Detailed Flow

### Phase 1: Local-First Constraint Application

#### Step 1.1: User Adds Memory on Device

```python
# Device SDK
paprdb = PaprDB(db_path="papr.db")

# User adds memory
memory = {
    "content": "Project Alpha is now completed",
    "type": "memory"
}

# SDK calls add_memory
result = await paprdb.add_memory(memory)
```

#### Step 1.2: Extract Nodes from Memory (LLM)

```python
# Device SDK (or cloud if LLM not available on device)
async def add_memory(memory: Dict):
    # Extract nodes using LLM (or call cloud API)
    extracted_nodes = await extract_nodes_from_memory(memory)
    # Returns: [{"type": "Project", "properties": {"name": "Alpha", "status": "completed"}}]
    
    # Apply constraints to each node
    for node in extracted_nodes:
        await self.apply_constraints_and_store(node)
```

#### Step 1.3: Apply Constraints Locally

```python
async def apply_constraints_and_store(self, node: Dict):
    """
    Apply node constraints locally before storing.
    This is the KEY step - constraints applied on device first.
    """
    node_type = node["type"]
    properties = node["properties"]
    
    # 1. Fetch constraints from local SQLite
    constraints = self._get_constraints_for_type(node_type)
    # Returns: [
    #   {
    #     "node_type": "Project",
    #     "when": {"priority": "high"},
    #     "force": {"workspace_id": "ws_123"},
    #     "merge": ["status"],
    #     "create": "auto",
    #     "search": {"mode": "semantic", "threshold": 0.85}
    #   }
    # ]
    
    # 2. Filter constraints by 'when' condition
    applicable_constraints = []
    for constraint in constraints:
        if self._matches_when(properties, constraint.get("when")):
            applicable_constraints.append(constraint)
    
    # 3. Apply 'force' (override values)
    for constraint in applicable_constraints:
        if constraint.get("force"):
            properties.update(constraint["force"])
    
    # 4. Search for existing node (if needed)
    existing_node = None
    if not constraint.get("node_id"):  # No direct node_id specified
        existing_node = await self._search_existing_node(
            node_type=node_type,
            properties=properties,
            search_config=constraint.get("search", {})
        )
    
    # 5. Handle creation policy
    if constraint.get("create") == "never" and not existing_node:
        # Skip creation (controlled vocabulary)
        logger.info(f"Skipping {node_type} creation (create: never)")
        return None
    
    # 6. Apply 'merge' (if existing found)
    if existing_node and constraint.get("merge"):
        for prop_name in constraint["merge"]:
            if prop_name in properties:
                existing_node["properties"][prop_name] = properties[prop_name]
        # Update existing node
        node_id = existing_node["id"]
        await self._update_node(node_id, existing_node["properties"])
        return node_id
    
    # 7. Create new node (with constraints applied)
    node_id = await self._create_node(
        node_type=node_type,
        properties=properties,  # Already has 'force' applied
        constraint_version=constraint.get("version", 1)
    )
    
    return node_id
```

#### Step 1.4: Store in Local SQLite

```python
async def _create_node(self, node_type: str, properties: Dict, constraint_version: int):
    """Store node in local SQLite with constraint metadata."""
    node_id = f"{node_type}_{int(time.time() * 1000)}"
    now = int(time.time())
    
    self.conn.execute(
        """
        INSERT INTO nodes (
            id, type, properties,
            created_at, updated_at,
            constraint_applied_at, constraint_version,
            sync_status, sync_pending
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            node_id, node_type, json.dumps(properties),
            now, now,
            now, constraint_version,
            "pending", True  # Mark for background sync
        )
    )
    
    return node_id
```

---

### Phase 2: Background Sync to Cloud

#### Step 2.1: Queue for Sync

```python
# After storing locally, add to sync queue
sync_queue.add({
    "node_id": node_id,
    "node_type": node_type,
    "properties": properties,
    "constraints_applied": True,
    "constraint_version": constraint_version,
    "sync_status": "pending",
    "created_at": now
})
```

#### Step 2.2: Background Sync Worker

```python
class BackgroundSyncWorker:
    """Background worker that syncs local changes to cloud."""
    
    async def run(self):
        """Run continuously, syncing pending items."""
        while True:
            try:
                # Get pending items
                pending = self.sync_queue.get_pending(limit=10)
                
                if not pending:
                    await asyncio.sleep(5)  # Wait 5 seconds
                    continue
                
                # Sync each item
                for item in pending:
                    await self.sync_item_to_cloud(item)
                    
            except Exception as e:
                logger.error(f"Background sync error: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def sync_item_to_cloud(self, item: Dict):
        """Sync single item to cloud."""
        try:
            # Call cloud API
            response = await httpx.post(
                "https://api.papr.ai/v1/memories",
                json={
                    "content": item.get("content"),
                    "extracted_nodes": [{
                        "type": item["node_type"],
                        "properties": item["properties"],
                        "constraints_applied": True,
                        "constraint_version": item["constraint_version"]
                    }]
                },
                headers={"X-API-Key": self.api_key}
            )
            
            if response.status_code == 200:
                # Mark as synced
                self.sync_queue.mark_synced(item["node_id"])
                self.conn.execute(
                    "UPDATE nodes SET sync_status = ?, sync_pending = ? WHERE id = ?",
                    ("synced", False, item["node_id"])
                )
            else:
                # Mark as failed (will retry)
                self.sync_queue.mark_failed(item["node_id"])
                
        except Exception as e:
            logger.error(f"Failed to sync {item['node_id']}: {e}")
            self.sync_queue.mark_failed(item["node_id"])
```

---

### Phase 3: Cloud Validation & Storage

#### Step 3.1: Cloud Receives Node

```python
# routers/v1/memory_routes_v1.py

@router.post("/memories")
async def add_memory(
    request: AddMemoryRequest,
    memory_graph: MemoryGraph = Depends(get_memory_graph)
):
    """
    Cloud endpoint receives memory with extracted nodes.
    Validates constraints and stores in Neo4j + Qdrant.
    """
    
    for extracted_node in request.extracted_nodes:
        # Check if constraints were already applied on device
        if extracted_node.get("constraints_applied"):
            # Validate constraint version matches cloud
            cloud_constraints = await get_constraints_for_type(
                extracted_node["type"],
                workspace_id=request.workspace_id
            )
            
            device_version = extracted_node.get("constraint_version", 1)
            cloud_version = cloud_constraints.get("version", 1)
            
            if device_version != cloud_version:
                # Re-apply constraints (cloud has newer version)
                logger.info(f"Re-applying constraints (device v{device_version} != cloud v{cloud_version})")
                extracted_node = await apply_constraints(
                    extracted_node,
                    cloud_constraints
                )
            else:
                # Device constraints match cloud - trust device
                logger.info("Device constraints match cloud - using device values")
        
        # Store in Neo4j + Qdrant
        await memory_graph.store_node(
            node_type=extracted_node["type"],
            properties=extracted_node["properties"],
            workspace_id=request.workspace_id
        )
```

---

### Phase 4: Constraint Definitions Sync

#### Step 4.1: Sync Constraint Definitions

```python
# routers/v1/sync_routes.py

@router.get("/constraints")
async def get_constraints(
    request: Request,
    since: Optional[str] = Query(None),  # ISO timestamp
    memory_graph: MemoryGraph = Depends(get_memory_graph)
):
    """
    Return constraint definitions for device sync.
    Device calls this to update local constraint cache.
    """
    
    # Get constraints from cloud (Parse Server or database)
    constraints = await get_user_constraints(
        user_id=resolved_user_id,
        workspace_id=resolved_workspace_id,
        since=since  # Only return updated constraints
    )
    
    return {
        "constraints": constraints,
        "version": get_latest_constraint_version(),
        "updated_at": datetime.now().isoformat()
    }
```

#### Step 4.2: Device Updates Constraint Cache

```python
# Device SDK

async def sync_constraints(self):
    """Sync constraint definitions from cloud."""
    # Get last sync timestamp
    last_sync = self.conn.execute(
        "SELECT MAX(updated_at) FROM node_constraints"
    ).fetchone()[0]
    
    # Call cloud API
    response = await httpx.get(
        f"https://api.papr.ai/v1/sync/constraints?since={last_sync}",
        headers={"X-API-Key": self.api_key}
    )
    
    constraints = response.json()["constraints"]
    
    # Update local SQLite
    for constraint in constraints:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO node_constraints (
                id, node_type, when_condition, force_properties,
                merge_properties, create_policy, search_config,
                version, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                constraint["id"],
                constraint["node_type"],
                json.dumps(constraint.get("when")),
                json.dumps(constraint.get("force")),
                json.dumps(constraint.get("merge")),
                constraint.get("create", "auto"),
                json.dumps(constraint.get("search", {})),
                constraint["version"],
                int(time.time())
            )
        )
```

---

### Phase 5: Delta Sync (Cloud → Device)

#### Step 5.1: Cloud Delta Endpoint

```python
# routers/v1/sync_routes.py

@router.get("/delta")
async def get_sync_delta(
    request: Request,
    cursor: Optional[str] = Query(None),
    include_constraints: bool = Query(True),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
):
    """
    Return nodes updated since cursor.
    Includes constraint changes if include_constraints=True.
    """
    
    # Decode cursor
    watermark_dt = _decode_cursor(cursor) if cursor else datetime.min
    
    # Fetch updated nodes from Neo4j
    updated_nodes = await fetch_nodes_since(
        workspace_id=resolved_workspace_id,
        since=watermark_dt
    )
    
    # Apply constraints to each node (if needed)
    items = []
    for node in updated_nodes:
        # Check if constraints need re-application
        if include_constraints:
            constraints = await get_constraints_for_type(node["type"])
            node = await apply_constraints(node, constraints)
        
        items.append({
            "id": node["id"],
            "type": node["type"],
            "properties": node["properties"],
            "action": "upsert",
            "constraints_applied": True,
            "constraint_version": constraints.get("version", 1),
            "updatedAt": node["updated_at"]
        })
    
    # Encode next cursor
    next_cursor = _encode_cursor(
        datetime.now(),
        items[-1]["id"] if items else ""
    )
    
    return {
        "items": items,
        "next_cursor": next_cursor,
        "has_more": len(items) == limit
    }
```

#### Step 5.2: Device Applies Delta

```python
# Device SDK

async def sync_delta(self, cursor: Optional[str] = None):
    """Sync delta updates from cloud."""
    # Call cloud API
    response = await httpx.get(
        f"https://api.papr.ai/v1/sync/delta?cursor={cursor}&include_constraints=true",
        headers={"X-API-Key": self.api_key}
    )
    
    delta = response.json()
    
    # Apply each update to local SQLite
    for item in delta["items"]:
        if item["action"] == "upsert":
            # Update or insert node
            self.conn.execute(
                """
                INSERT OR REPLACE INTO nodes (
                    id, type, properties,
                    updated_at, constraint_applied_at, constraint_version
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    item["id"],
                    item["type"],
                    json.dumps(item["properties"]),
                    int(datetime.fromisoformat(item["updatedAt"]).timestamp()),
                    int(time.time()),
                    item.get("constraint_version", 1)
                )
            )
    
    # Return next cursor for next sync
    return delta["next_cursor"]
```

---

### Phase 6: GraphQL Queries with Custom Ontology

#### Step 6.1: GraphQL Query (Local)

```python
# Device SDK

def graphql_query(self, query: str, variables: Dict = None):
    """
    Execute GraphQL query against local SQLite.
    Uses custom ontology from local schema.
    """
    # Parse GraphQL query
    ast = parse_graphql(query)
    
    # Get schema from local SQLite
    schema = self._get_local_schema()
    
    # Translate GraphQL to SQL (using custom ontology)
    sql = self._graphql_to_sql(ast, schema)
    
    # Execute SQL
    results = self.conn.execute(sql).fetchall()
    
    # Format as GraphQL response
    return self._format_graphql_response(results, ast)
```

#### Step 6.2: GraphQL → SQL Translation (with Custom Ontology)

```python
def _graphql_to_sql(self, ast: GraphQLAST, schema: Dict) -> str:
    """
    Translate GraphQL query to SQL using custom ontology.
    """
    # Example GraphQL query:
    # query {
    #   project(id: "123") {
    #     name
    #     tasks {
    #       title
    #     }
    #   }
    # }
    
    # Get node type from schema (custom ontology)
    node_type = ast.selection_set[0].name  # "project"
    schema_node = schema["node_types"].get(node_type)
    
    # Build SQL with recursive CTE
    sql = f"""
    WITH RECURSIVE {node_type}_traversal AS (
        -- Start from root node
        SELECT 
            n.id,
            n.type,
            n.properties,
            0 as depth
        FROM nodes n
        WHERE n.type = '{schema_node["label"]}'  -- Use schema label
          AND json_extract(n.properties, '$.id') = '123'
        
        UNION ALL
        
        -- Traverse relationships (from custom ontology)
        SELECT 
            n2.id,
            n2.type,
            n2.properties,
            pt.depth + 1
        FROM {node_type}_traversal pt
        JOIN edges e ON e.source_id = pt.id
        JOIN nodes n2 ON n2.id = e.target_id
        WHERE e.type = '{schema_node["relationships"]["tasks"]["type"]}'  -- From schema
          AND pt.depth < 2
    )
    SELECT 
        json_extract(properties, '$.name') as name,
        json_extract(properties, '$.title') as title
    FROM {node_type}_traversal;
    """
    
    return sql
```

---

## 🗄️ Database Schema

### SQLite Schema (Device)

```sql
-- Nodes table
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    properties JSON NOT NULL,
    embedding BLOB,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    
    -- Constraint metadata
    constraint_applied_at INTEGER,
    constraint_version INTEGER,
    
    -- Sync metadata
    sync_status TEXT DEFAULT 'pending',  -- pending, synced, failed
    sync_pending BOOLEAN DEFAULT TRUE,
    sync_attempts INTEGER DEFAULT 0,
    
    INDEX idx_nodes_type (type),
    INDEX idx_nodes_sync (sync_pending, sync_status)
);

-- Node constraints table (synced from cloud)
CREATE TABLE node_constraints (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    
    -- Constraint definition
    when_condition JSON,      -- 'when' field
    force_properties JSON,    -- 'force' field
    merge_properties JSON,    -- 'merge' field
    create_policy TEXT,       -- 'create' field ("auto" or "never")
    search_config JSON,       -- 'search' field
    
    -- Metadata
    workspace_id TEXT,
    version INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    
    INDEX idx_constraints_type (node_type),
    INDEX idx_constraints_workspace (workspace_id)
);

-- Sync queue table
CREATE TABLE sync_queue (
    id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    properties JSON NOT NULL,
    constraints_applied BOOLEAN DEFAULT FALSE,
    constraint_version INTEGER,
    sync_status TEXT DEFAULT 'pending',
    created_at INTEGER NOT NULL,
    sync_attempts INTEGER DEFAULT 0,
    
    FOREIGN KEY (node_id) REFERENCES nodes(id),
    INDEX idx_sync_queue_status (sync_status, created_at)
);
```

---

## 🔄 Sync Strategy Summary

### 1. **Local-First (Device)**
- ✅ Constraints applied immediately on device
- ✅ Node stored in local SQLite
- ✅ User can query immediately (offline)

### 2. **Background Sync (Device → Cloud)**
- ✅ Queued for background sync
- ✅ Worker syncs to cloud asynchronously
- ✅ Retry on failure

### 3. **Cloud Validation**
- ✅ Cloud validates constraint version
- ✅ Re-applies if version mismatch
- ✅ Stores in Neo4j + Qdrant

### 4. **Constraint Definitions Sync (Cloud → Device)**
- ✅ Device syncs constraint definitions periodically
- ✅ Updates local constraint cache
- ✅ Future nodes use updated constraints

### 5. **Delta Sync (Cloud → Device)**
- ✅ Device pulls updates from cloud
- ✅ Applies constraint changes
- ✅ Updates local SQLite

### 6. **GraphQL Queries (Local)**
- ✅ Uses custom ontology from local schema
- ✅ Works offline
- ✅ Sub-100ms performance

---

## ✅ Benefits of This Architecture

1. **Offline-First**: Constraints applied locally, works without internet
2. **Fast UX**: Immediate feedback, background sync doesn't block
3. **Consistency**: Cloud validates and re-applies if needed
4. **Flexibility**: Custom ontology for GraphQL queries
5. **Resilience**: Retry logic, queue persistence

---

## 🎯 Next Steps

1. **Implement Local Constraint Application** (Step 1.3)
2. **Build Background Sync Worker** (Step 2.2)
3. **Add Constraint Definitions Sync** (Step 4.1)
4. **Extend Delta Sync** (Step 5.1)
5. **GraphQL with Custom Ontology** (Step 6.1)

This architecture provides a robust, offline-first solution for node constraints with paprDB! 🚀

