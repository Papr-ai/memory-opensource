# paprDB Architecture: Cloud + On-Device

## Executive Summary

**Recommended Architecture**:
- **Cloud**: Keep Qdrant + Neo4j (for scale, already working)
- **On-Device**: SQLite paprDB (for offline GraphQL, small subgraph)

**Why This Works**:
- ✅ Cloud handles scale (Qdrant for vectors, Neo4j for graph)
- ✅ Device gets predicted subgraph (only relevant nodes)
- ✅ SQLite perfect for small datasets (<100K nodes on device)
- ✅ GraphQL works offline (sub-100ms queries)
- ✅ No need to replace cloud infrastructure

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    CLOUD (Production)                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Qdrant     │    │    Neo4j     │    │  Parse Server │ │
│  │  (Vectors)   │    │   (Graph)    │    │   (Metadata)  │ │
│  │              │    │              │    │               │ │
│  │ 100M+ nodes  │    │ 100M+ nodes  │    │ 100M+ nodes   │ │
│  │ Fast search  │    │ Graph queries│    │ ACL, schemas  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                   │                   │            │
│         └───────────────────┼───────────────────┘            │
│                             │                                │
│                    ┌────────▼────────┐                       │
│                    │  Sync Service   │                       │
│                    │  (Predicts      │                       │
│                    │   Subgraph)     │                       │
│                    └────────┬────────┘                       │
│                             │                                │
└─────────────────────────────┼────────────────────────────────┘
                               │
                               │ Sync Predicted Subgraph
                               │ (Only relevant nodes/edges)
                               │
┌───────────────────────────────▼──────────────────────────────┐
│                  ON-DEVICE (SQLite paprDB)                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              SQLite Database                          │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │ │
│  │  │    Nodes     │  │    Edges     │  │ Constraints │ │ │
│  │  │  (10K-100K)  │  │  (50K-500K)  │  │  (Pre-applied)│ │ │
│  │  │              │  │              │  │              │ │ │
│  │  │ + Embeddings │  │ + Types      │  │ + Schemas    │ │ │
│  │  └──────────────┘  └──────────────┘  └────────────┘ │ │
│  │                                                       │ │
│  │  ┌──────────────────────────────────────────────┐   │ │
│  │  │        GraphQL Engine                        │   │ │
│  │  │  - GraphQL → SQL translation                 │   │ │
│  │  │  - Recursive CTEs for traversal             │   │ │
│  │  │  - Sub-100ms queries                         │   │ │
│  │  │  - Works OFFLINE                             │   │ │
│  │  └──────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Why SQLite is Perfect for On-Device

### 1. **Small Dataset** (<100K nodes)

**Cloud**: 100M+ nodes (Qdrant + Neo4j handle this)
**Device**: 10K-100K nodes (predicted subgraph)

**SQLite Performance**:
- ✅ 10K nodes: <1ms queries
- ✅ 100K nodes: <10ms queries
- ✅ 1M nodes: <100ms queries (still acceptable)

**Verdict**: SQLite handles device dataset easily.

---

### 2. **Offline-First Use Case**

**Requirements**:
- ✅ Works without internet
- ✅ Fast queries (sub-100ms)
- ✅ GraphQL support
- ✅ Node constraints (pre-applied from cloud)

**SQLite Delivers**:
- ✅ Zero-config (built into OS)
- ✅ ACID transactions
- ✅ SQL native (GraphQL translation easy)
- ✅ Small file size (~10-50MB for 100K nodes)

---

### 3. **Sync Strategy: Predicted Subgraph**

**How It Works**:

```python
# Cloud: Predict what user needs
subgraph = predict_subgraph(
    goals=["Help customer with order"],
    tasks=["Query recent orders", "Check shipping status"],
    user_id=user_id,
    workspace_id=workspace_id
)

# Returns: Only relevant nodes/edges
{
    "nodes": ["customer_123", "order_456", "order_789", "product_abc"],
    "edges": [
        {"source": "customer_123", "target": "order_456", "type": "PURCHASED"},
        {"source": "order_456", "target": "product_abc", "type": "CONTAINS"}
    ],
    "size_bytes": 2400000,  # ~2.4MB
    "confidence": 0.92
}

# Sync to device
paprdb.sync_subgraph(subgraph)  # Creates/updates SQLite file

# Device: Query locally (offline)
result = paprdb.graphql_query("""
    query {
        customer(id: "123") {
            orders {
                status
                products {
                    name
                }
            }
        }
    }
""")  # <100ms, works offline
```

**Benefits**:
- ✅ Only sync what's needed (not full graph)
- ✅ Small file size (2-10MB typical)
- ✅ Fast sync (seconds, not minutes)
- ✅ Offline queries work immediately

---

## Architecture Decision: Cloud vs Device

### Option 1: Keep Cloud Stack, Add Device SQLite (RECOMMENDED)

```
Cloud:
  - Qdrant (vectors) ✅ Keep
  - Neo4j (graph) ✅ Keep
  - Parse Server (metadata) ✅ Keep
  
Device:
  - SQLite paprDB (NEW) ✅ Add
    - Sync predicted subgraph from cloud
    - GraphQL queries offline
    - Node constraints pre-applied
```

**Pros**:
- ✅ No changes to cloud (already working)
- ✅ Device gets offline capability
- ✅ Best of both worlds

**Cons**:
- ⚠️ Need to sync cloud → device (but you already do this with /v1/sync/tiers)

**Verdict**: ✅ **Recommended** - Minimal changes, maximum benefit

---

### Option 2: Replace Cloud Stack with paprDB (NOT RECOMMENDED)

```
Cloud:
  - paprDB (RocksDB backend) ❌ Replace Qdrant/Neo4j
  
Device:
  - SQLite paprDB ✅
```

**Pros**:
- ✅ Unified codebase (cloud + device)
- ✅ Single database to maintain

**Cons**:
- ❌ Replace working infrastructure (risky)
- ❌ Need to migrate 100M+ nodes
- ❌ RocksDB adds complexity
- ❌ May lose performance (Qdrant/Neo4j optimized)

**Verdict**: ❌ **Not Recommended** - Too risky, not enough benefit

---

## Implementation Plan

### Phase 1: On-Device SQLite paprDB (3 months)

**Goal**: Enable offline GraphQL queries on device

**Tasks**:
1. **SQLite Schema** (2 weeks)
   - Nodes, edges, constraints, schemas tables
   - Vector index (sqlite-vec)
   - Graph indexes

2. **GraphQL Engine** (3 weeks)
   - GraphQL → SQL translation
   - Recursive CTEs for traversal
   - Query optimization

3. **Sync Integration** (2 weeks)
   - Extend `/v1/sync/tiers` to return subgraph
   - Add `/v1/sync/subgraph` endpoint
   - Device SDK sync logic

4. **Node Constraints** (1 week)
   - Pre-apply constraints in cloud
   - Sync to device (already applied)
   - Device just queries (no constraint logic)

5. **Testing** (2 weeks)
   - GraphQL query tests
   - Performance benchmarks
   - Offline scenarios

**Timeline**: 3 months

---

### Phase 2: Cloud Integration (Optional, 1 month)

**Goal**: Use paprDB for cloud sync operations (not replacing Qdrant/Neo4j)

**Tasks**:
1. **Sync Service Enhancement**
   - Predict subgraph (already have Tier0PredictiveBuilder)
   - Apply constraints (already have constraint logic)
   - Generate SQLite file
   - Return to device

2. **Delta Sync**
   - Track changes in cloud
   - Sync only updates to device
   - Incremental SQLite updates

**Timeline**: 1 month

---

## Code Example: Sync Subgraph Endpoint

```python
# routers/v1/sync_routes.py

@router.post("/subgraph", response_model=SyncSubgraphResponse)
async def get_sync_subgraph(
    request: Request,
    response: Response,
    sync_request: SyncSubgraphRequest = Body(...),
    # ... auth ...
    memory_graph: MemoryGraph = Depends(get_memory_graph)
) -> SyncSubgraphResponse:
    """
    Return predicted subgraph for device sync.
    
    This extends /v1/sync/tiers to include:
    - Graph relationships (edges)
    - Node constraints (pre-applied)
    - Schema definitions
    """
    
    # 1. Predict subgraph (reuse Tier0PredictiveBuilder)
    predictive_builder = Tier0PredictiveBuilder()
    tier0_nodes = await predictive_builder.build(
        user_id=resolved_user_id,
        session_token=sessionToken,
        workspace_id=resolved_workspace_id,
        max_items=sync_request.max_nodes,
        memory_graph=memory_graph,
        # ... other params ...
    )
    
    # 2. Get graph relationships for these nodes
    node_ids = [node.id for node in tier0_nodes]
    edges = await memory_graph.get_edges_for_nodes(
        node_ids=node_ids,
        max_depth=2,  # Include 2-hop relationships
        session_token=sessionToken
    )
    
    # 3. Apply node constraints (pre-apply in cloud)
    constraints = await get_user_constraints(resolved_user_id, resolved_workspace_id)
    constrained_nodes = []
    for node in tier0_nodes:
        constrained_node = apply_constraints(node, constraints)
        constrained_nodes.append(constrained_node)
    
    # 4. Get schema definitions
    schema = await get_user_schema(resolved_user_id, resolved_workspace_id)
    
    # 5. Return subgraph (device will create SQLite file)
    return SyncSubgraphResponse(
        nodes=constrained_nodes,
        edges=edges,
        constraints=constraints,  # For reference (already applied)
        schema=schema,
        size_bytes=calculate_size(constrained_nodes, edges),
        ttl_seconds=3600  # Cache for 1 hour
    )
```

---

## Device SDK Example

```python
# Device SDK (Python/TypeScript)

class PaprDB:
    def __init__(self, db_path: str = "papr.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()
    
    async def sync(self, api_key: str):
        """Sync predicted subgraph from cloud."""
        # Call /v1/sync/subgraph
        response = await httpx.post(
            "https://api.papr.ai/v1/sync/subgraph",
            json={
                "goals": self.goals,
                "tasks": self.tasks,
                "max_nodes": 10000
            },
            headers={"X-API-Key": api_key}
        )
        subgraph = response.json()
        
        # Store in SQLite
        self._store_subgraph(subgraph)
    
    def query(self, graphql_query: str) -> dict:
        """Query locally (offline)."""
        sql = self._graphql_to_sql(graphql_query)
        results = self.conn.execute(sql).fetchall()
        return self._format_results(results)
    
    def _store_subgraph(self, subgraph: dict):
        """Store nodes, edges, constraints in SQLite."""
        with self.conn:
            # Store nodes (constraints already applied)
            for node in subgraph['nodes']:
                self.conn.execute(
                    "INSERT OR REPLACE INTO nodes (id, type, properties, embedding) VALUES (?, ?, ?, ?)",
                    (node['id'], node['type'], json.dumps(node['properties']), node.get('embedding'))
                )
            
            # Store edges
            for edge in subgraph['edges']:
                self.conn.execute(
                    "INSERT OR REPLACE INTO edges (id, source_id, target_id, type) VALUES (?, ?, ?, ?)",
                    (edge['id'], edge['source_id'], edge['target_id'], edge['type'])
                )
            
            # Store schema (for GraphQL schema generation)
            if 'schema' in subgraph:
                self.conn.execute(
                    "INSERT OR REPLACE INTO schemas (id, name, node_types, relationship_types) VALUES (?, ?, ?, ?)",
                    (subgraph['schema']['id'], subgraph['schema']['name'],
                     json.dumps(subgraph['schema']['node_types']),
                     json.dumps(subgraph['schema']['relationship_types']))
                )

# Usage
paprdb = PaprDB()

# Sync from cloud (when online)
await paprdb.sync(api_key="your_key")

# Query locally (works offline)
results = paprdb.query("""
    query {
        project(id: "123") {
            name
            tasks {
                title
                status
            }
        }
    }
""")
```

---

## Performance Expectations

### On-Device (SQLite)

| Dataset Size | Query Time | File Size |
|--------------|------------|-----------|
| 1K nodes | <1ms | ~100KB |
| 10K nodes | <5ms | ~1MB |
| 100K nodes | <50ms | ~10MB |
| 1M nodes | <500ms | ~100MB |

**Typical Device Subgraph**: 10K-100K nodes → **<50ms queries, ~10MB file**

### Cloud (Qdrant + Neo4j)

| Operation | Performance |
|-----------|-------------|
| Vector search | <50ms (Qdrant) |
| Graph query | <100ms (Neo4j) |
| Subgraph prediction | <200ms (Tier0PredictiveBuilder) |
| Sync generation | <500ms (total) |

---

## Benefits Summary

### For Developers

✅ **Offline Support**: GraphQL queries work without internet
✅ **Fast Queries**: Sub-100ms on device (no network latency)
✅ **Same API**: GraphQL works same on cloud and device
✅ **Privacy**: Data stays on device

### For Your Product

✅ **Differentiation**: No competitor has offline GraphQL
✅ **User Experience**: Works in airplane mode
✅ **Reduced Load**: Device queries don't hit cloud
✅ **Scalability**: Cloud handles scale, device handles UX

---

## Conclusion

**SQLite is perfect for on-device paprDB** because:

1. ✅ **Small Dataset**: Device only has predicted subgraph (10K-100K nodes)
2. ✅ **Offline-First**: SQLite works everywhere, zero-config
3. ✅ **GraphQL Native**: SQL makes GraphQL translation trivial
4. ✅ **No Cloud Changes**: Keep Qdrant + Neo4j (already working)
5. ✅ **Fast Development**: 3 months vs 6+ months for RocksDB

**Architecture**:
- **Cloud**: Qdrant + Neo4j (scale, performance) ✅ Keep
- **Device**: SQLite paprDB (offline, GraphQL) ✅ Add

**Next Step**: Build on-device SQLite paprDB, sync predicted subgraph from cloud.

