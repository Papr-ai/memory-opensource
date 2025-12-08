# paprDB Prototype: SQLite Schema Design

## Quick Start Prototype

This is a minimal prototype to validate the paprDB concept. It shows how to combine vector search, graph relationships, and node constraints in a single SQLite database.

---

## Schema Design

```sql
-- Enable required SQLite extensions
-- Note: This requires sqlite-vec or custom HNSW extension for vector search

-- ============================================
-- NODES TABLE (with vector embeddings)
-- ============================================
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    properties JSON NOT NULL,
    
    -- Vector embedding (stored as BLOB)
    embedding BLOB,
    embedding_dim INTEGER,
    
    -- Metadata
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    workspace_id TEXT,
    user_id TEXT,
    
    -- Constraint tracking
    constraint_applied_at INTEGER,
    constraint_version INTEGER,
    
    -- Indexes
    INDEX idx_nodes_type (type),
    INDEX idx_nodes_workspace (workspace_id),
    INDEX idx_nodes_user (user_id),
    INDEX idx_nodes_updated (updated_at)
);

-- ============================================
-- EDGES TABLE (graph relationships)
-- ============================================
CREATE TABLE edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    type TEXT NOT NULL,
    properties JSON,
    created_at INTEGER NOT NULL,
    
    FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE,
    
    -- Indexes for fast traversal
    INDEX idx_edges_source (source_id),
    INDEX idx_edges_target (target_id),
    INDEX idx_edges_type (type),
    INDEX idx_edges_source_type (source_id, type),
    INDEX idx_edges_target_type (target_id, type)
);

-- ============================================
-- NODE CONSTRAINTS TABLE
-- ============================================
CREATE TABLE node_constraints (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    
    -- Constraint definition
    match_condition JSON,  -- WHEN to apply (e.g., {"priority": "high"})
    set_properties JSON,    -- FORCE these values (e.g., {"team": "backend"})
    update_properties JSON, -- UPDATE these from AI (e.g., ["status", "priority"])
    
    -- Metadata
    workspace_id TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    
    INDEX idx_constraints_type (node_type),
    INDEX idx_constraints_workspace (workspace_id)
);

-- ============================================
-- SCHEMAS TABLE (custom ontology)
-- ============================================
CREATE TABLE schemas (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    
    -- Schema definition
    node_types JSON,         -- User-defined node types
    relationship_types JSON, -- User-defined relationship types
    
    -- Metadata
    workspace_id TEXT,
    version INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    
    INDEX idx_schemas_workspace (workspace_id)
);

-- ============================================
-- VECTOR INDEX (using sqlite-vec or HNSW)
-- ============================================
-- Option 1: Using sqlite-vec extension
CREATE VIRTUAL TABLE vector_index USING vec0(
    node_id TEXT,
    embedding vector(768)  -- Adjust dimension as needed
);

-- Option 2: Using custom HNSW index (if sqlite-vec not available)
-- Store vectors in nodes.embedding, index separately
-- This requires custom C extension or Python wrapper

-- ============================================
-- HELPER VIEWS
-- ============================================

-- View: Nodes with their constraint status
CREATE VIEW nodes_with_constraints AS
SELECT 
    n.*,
    nc.id as constraint_id,
    nc.set_properties as constraint_set,
    nc.update_properties as constraint_update
FROM nodes n
LEFT JOIN node_constraints nc ON n.type = nc.node_type
WHERE nc.workspace_id = n.workspace_id OR nc.workspace_id IS NULL;

-- View: Graph statistics
CREATE VIEW graph_stats AS
SELECT 
    type,
    COUNT(*) as node_count,
    COUNT(DISTINCT workspace_id) as workspace_count
FROM nodes
GROUP BY type;
```

---

## Core Operations

### 1. Create Node (with Constraints)

```python
import sqlite3
import json
from typing import Dict, List, Optional
import time

def create_node(
    conn: sqlite3.Connection,
    node_type: str,
    properties: Dict,
    embedding: Optional[List[float]] = None,
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> str:
    """
    Create a node with automatic constraint application.
    """
    node_id = f"{node_type}_{int(time.time() * 1000)}"
    now = int(time.time())
    
    # 1. Fetch applicable constraints
    constraints = conn.execute(
        """
        SELECT match_condition, set_properties, update_properties
        FROM node_constraints
        WHERE node_type = ? AND (workspace_id = ? OR workspace_id IS NULL)
        """,
        (node_type, workspace_id)
    ).fetchall()
    
    # 2. Apply constraints
    for match_cond, set_props, update_props in constraints:
        # Check if match condition applies
        if match_cond and not _matches_condition(properties, json.loads(match_cond)):
            continue
        
        # Apply 'set' properties (force values)
        if set_props:
            set_dict = json.loads(set_props)
            properties.update(set_dict)
        
        # Apply 'update' properties (if node exists)
        if update_props:
            update_list = json.loads(update_props)
            existing = _find_existing_node(conn, node_type, properties)
            if existing:
                for prop in update_list:
                    if prop in properties:
                        existing_props = json.loads(existing['properties'])
                        existing_props[prop] = properties[prop]
                        # Update existing node
                        conn.execute(
                            "UPDATE nodes SET properties = ?, updated_at = ? WHERE id = ?",
                            (json.dumps(existing_props), now, existing['id'])
                        )
                        return existing['id']
    
    # 3. Insert new node
    embedding_blob = _embedding_to_blob(embedding) if embedding else None
    embedding_dim = len(embedding) if embedding else None
    
    conn.execute(
        """
        INSERT INTO nodes (
            id, type, properties, embedding, embedding_dim,
            created_at, updated_at, workspace_id, user_id,
            constraint_applied_at, constraint_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            node_id, node_type, json.dumps(properties),
            embedding_blob, embedding_dim,
            now, now, workspace_id, user_id,
            now, 1
        )
    )
    
    # 4. Update vector index
    if embedding:
        conn.execute(
            "INSERT INTO vector_index (node_id, embedding) VALUES (?, ?)",
            (node_id, embedding)
        )
    
    return node_id

def _matches_condition(properties: Dict, condition: Dict) -> bool:
    """Check if properties match the condition."""
    for key, value in condition.items():
        if properties.get(key) != value:
            return False
    return True

def _find_existing_node(conn: sqlite3.Connection, node_type: str, properties: Dict) -> Optional[Dict]:
    """Find existing node by type and key properties."""
    # Simple implementation: match on 'name' or 'id' if present
    if 'name' in properties:
        row = conn.execute(
            "SELECT * FROM nodes WHERE type = ? AND json_extract(properties, '$.name') = ?",
            (node_type, properties['name'])
        ).fetchone()
        if row:
            return dict(zip([col[0] for col in conn.execute("PRAGMA table_info(nodes)").fetchall()], row))
    return None

def _embedding_to_blob(embedding: List[float]) -> bytes:
    """Convert embedding list to BLOB."""
    import struct
    return struct.pack(f'{len(embedding)}f', *embedding)
```

### 2. Vector Search

```python
def vector_search(
    conn: sqlite3.Connection,
    query_embedding: List[float],
    top_k: int = 10,
    node_type: Optional[str] = None,
    workspace_id: Optional[str] = None
) -> List[Dict]:
    """
    Vector similarity search using sqlite-vec.
    """
    # Build filter
    filters = []
    if node_type:
        filters.append(f"n.type = '{node_type}'")
    if workspace_id:
        filters.append(f"n.workspace_id = '{workspace_id}'")
    filter_clause = " AND " + " AND ".join(filters) if filters else ""
    
    # Vector search query (using sqlite-vec syntax)
    query = f"""
    SELECT 
        n.id,
        n.type,
        n.properties,
        vec_distance_cosine(vi.embedding, ?) as similarity
    FROM vector_index vi
    JOIN nodes n ON n.id = vi.node_id
    WHERE 1=1 {filter_clause}
    ORDER BY similarity DESC
    LIMIT ?
    """
    
    results = conn.execute(query, (query_embedding, top_k)).fetchall()
    return [
        {
            'id': row[0],
            'type': row[1],
            'properties': json.loads(row[2]),
            'similarity': row[3]
        }
        for row in results
    ]
```

### 3. Graph Traversal (Recursive CTE)

```python
def graph_query(
    conn: sqlite3.Connection,
    start_node_id: str,
    edge_type: Optional[str] = None,
    max_depth: int = 3
) -> List[Dict]:
    """
    Graph traversal using recursive CTE.
    """
    edge_filter = f"AND e.type = '{edge_type}'" if edge_type else ""
    
    query = f"""
    WITH RECURSIVE graph_path AS (
        -- Base case: start node
        SELECT 
            n.id,
            n.type,
            n.properties,
            0 as depth,
            CAST(n.id AS TEXT) as path
        FROM nodes n
        WHERE n.id = ?
        
        UNION ALL
        
        -- Recursive case: traverse edges
        SELECT 
            n2.id,
            n2.type,
            n2.properties,
            gp.depth + 1,
            gp.path || ' -> ' || n2.id
        FROM graph_path gp
        JOIN edges e ON e.source_id = gp.id {edge_filter}
        JOIN nodes n2 ON n2.id = e.target_id
        WHERE gp.depth < ?
    )
    SELECT * FROM graph_path;
    """
    
    results = conn.execute(query, (start_node_id, max_depth)).fetchall()
    return [
        {
            'id': row[0],
            'type': row[1],
            'properties': json.loads(row[2]),
            'depth': row[3],
            'path': row[4]
        }
        for row in results
    ]
```

### 4. GraphQL Query Translation

```python
def graphql_to_sql(graphql_query: str) -> str:
    """
    Simple GraphQL to SQL translator (proof of concept).
    
    Example:
    query {
        project(id: "123") {
            name
            tasks {
                title
            }
        }
    }
    
    Translates to:
    """
    # This is a simplified example - full implementation would use a GraphQL parser
    
    sql = """
    WITH RECURSIVE project_tasks AS (
        -- Start from project
        SELECT 
            n.id,
            n.type,
            n.properties,
            0 as depth
        FROM nodes n
        WHERE n.type = 'Project' AND json_extract(n.properties, '$.id') = '123'
        
        UNION ALL
        
        -- Traverse to tasks
        SELECT 
            n2.id,
            n2.type,
            n2.properties,
            pt.depth + 1
        FROM project_tasks pt
        JOIN edges e ON e.source_id = pt.id
        JOIN nodes n2 ON n2.id = e.target_id
        WHERE e.type = 'HAS_TASK' AND pt.depth < 1
    )
    SELECT 
        json_extract(properties, '$.name') as name,
        json_extract(properties, '$.title') as title
    FROM project_tasks
    WHERE type IN ('Project', 'Task');
    """
    
    return sql
```

---

## Example Usage

```python
import sqlite3

# Initialize database
conn = sqlite3.connect('papr.db')
conn.execute('PRAGMA foreign_keys = ON')  # Enable foreign keys

# Create schema
# (Run the SQL schema from above)

# 1. Create a constraint
conn.execute(
    """
    INSERT INTO node_constraints (id, node_type, set_properties, workspace_id, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?)
    """,
    (
        'constraint_1',
        'Project',
        json.dumps({'team': 'backend', 'workspace_id': 'ws_123'}),
        'ws_123',
        int(time.time()),
        int(time.time())
    )
)

# 2. Create a node (constraints applied automatically)
node_id = create_node(
    conn,
    node_type='Project',
    properties={'name': 'Alpha', 'status': 'active'},
    workspace_id='ws_123',
    user_id='user_123'
)
# Result: Node created with team='backend' and workspace_id='ws_123' (from constraint)

# 3. Create an edge
conn.execute(
    """
    INSERT INTO edges (id, source_id, target_id, type, created_at)
    VALUES (?, ?, ?, ?, ?)
    """,
    ('edge_1', node_id, 'task_1', 'HAS_TASK', int(time.time()))
)

# 4. Vector search
results = vector_search(
    conn,
    query_embedding=[0.1, 0.2, 0.3, ...],  # Your query embedding
    top_k=10,
    node_type='Project'
)

# 5. Graph traversal
related = graph_query(
    conn,
    start_node_id=node_id,
    edge_type='HAS_TASK',
    max_depth=2
)

conn.commit()
```

---

## Performance Considerations

### Indexes

The schema includes indexes for:
- Node type lookups
- Workspace filtering
- Edge traversal (source/target)
- Vector search (via sqlite-vec)

### Optimization Tips

1. **Embedding Storage**: Use INT8 quantization for smaller size (4x reduction)
2. **JSON Properties**: Consider extracting frequently queried fields to columns
3. **Batch Operations**: Use transactions for bulk inserts
4. **Vector Index**: Use HNSW for better performance (requires custom extension)

### Scale Limits

- **SQLite**: Good for <10M nodes, <100M edges
- **For Scale**: Use hybrid approach (cloud Neo4j, device paprDB)

---

## Next Steps

1. **Prototype** (1 week)
   - Implement basic CRUD operations
   - Test vector search with sqlite-vec
   - Test graph traversal with recursive CTEs

2. **Benchmark** (1 week)
   - Compare vs Qdrant (vector search)
   - Compare vs Neo4j (graph queries)
   - Test with real data (10K nodes)

3. **Decide** (based on results)
   - If performance acceptable → Full build
   - If not → Hybrid approach

---

## References

- [sqlite-vec](https://github.com/asg017/sqlite-vec) - Vector search extension for SQLite
- [SQLite Recursive CTEs](https://www.sqlite.org/lang_with.html) - Graph traversal
- [ChromaDB Architecture](https://docs.trychroma.com/) - Inspiration for vector storage

