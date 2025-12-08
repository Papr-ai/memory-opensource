# Schema Systems Flow Diagram

## 🔄 **Current Auto-Discovery Flow (Active)**

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. User Uploads Document                                             │
│    POST /v1/documents/upload                                         │
│    - No schema_id provided                                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. LLM Generates Memories (Unconstrained)                            │
│    - LLM decides node types: "Security Protocol", "Agent", etc.      │
│    - LLM decides relationships: "PERFORMS", "MITIGATES", etc.        │
│    - NO schema enforcement                                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. Nodes/Relationships Created in Neo4j                              │
│    CREATE (n:SecurityProtocol {content: "..."})-[:MITIGATES]->(r:Risk)│
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. Pattern Discovery (_discover_neo4j_patterns_for_cache)            │
│    MATCH (source)-[rel]->(target)                                    │
│    WHERE source.user_id = $user_id                                   │
│    RETURN labels(source), type(rel), labels(target), count(*)       │
│                                                                       │
│    Result: [                                                         │
│      {source: "SecurityProtocol", rel: "MITIGATES", target: "Risk", count: 5},│
│      {source: "Agent", rel: "PERFORMS", target: "Step", count: 12}  │
│    ]                                                                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. Cache Patterns in Parse (ActiveNodeRel)                           │
│    POST /parse/classes/ActiveNodeRel                                 │
│    {                                                                 │
│      user: {Pointer to _User},                                       │
│      workspace: {Pointer to WorkSpace},                              │
│      activePatterns: "[{...top 50 patterns...}]"                     │
│    }                                                                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. Next Document Upload (Guided by Cache)                            │
│    - Fetch cached patterns from Parse                                │
│    - Inject patterns into LLM context (SOFT guidance)                │
│    - LLM tends to reuse: "SecurityProtocol", "Agent", "Risk"         │
│    - But can still invent new types!                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 **Desired Custom Schema Flow (Not Implemented)**

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Developer Creates Custom Schema                                   │
│    POST /v1/schemas                                                  │
│    {                                                                 │
│      name: "Security & Workflows",                                   │
│      node_types: {                                                   │
│        "SecurityBehavior": {...},                                    │
│        "Risk": {...},                                                │
│        "Impact": {...}                                               │
│      },                                                              │
│      relationship_types: {                                           │
│        "INTRODUCES_RISK": {source: "SecurityBehavior", target: "Risk"}│
│      }                                                               │
│    }                                                                 │
│    Returns: {schema_id: "abc123"}                                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. User Uploads Document WITH schema_id                              │
│    POST /v1/documents/upload                                         │
│    - file: "two-factor-authentication.pdf"                           │
│    - schema_id: "abc123" ❌ NOT CURRENTLY SUPPORTED                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. Workflow Receives schema_id                                       │
│    DocumentProcessingWorkflow.run(                                   │
│      upload_id=...,                                                  │
│      schema_id="abc123" ❌ NOT PASSED                                │
│    )                                                                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. Fetch Custom Schema from Parse                                    │
│    GET /parse/classes/UserGraphSchema/abc123                         │
│    Returns: {                                                        │
│      node_types: {...},                                              │
│      relationship_types: {...}                                       │
│    }                                                                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. LLM Generates with STRICT Schema Enforcement                      │
│    Prompt:                                                           │
│    "ONLY use these nodes: SecurityBehavior, Risk, Impact             │
│     ONLY use these relationships: INTRODUCES_RISK, CAUSES            │
│     ANY other types are INVALID"                                     │
│                                                                       │
│    LLM Output (validated):                                           │
│    {                                                                 │
│      nodes: [                                                        │
│        {label: "SecurityBehavior", content: "2FA"},                  │
│        {label: "Risk", content: "Account Takeover"}                  │
│      ],                                                              │
│      relationships: [                                                │
│        {type: "INTRODUCES_RISK", source: "2FA", target: "Account Takeover"}│
│      ]                                                               │
│    }                                                                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. Schema Validation Before Neo4j Creation                           │
│    for node in nodes:                                                │
│        if node.label not in schema.node_types:                       │
│            raise SchemaValidationError                               │
│                                                                       │
│    ✅ All nodes match schema → Proceed to Neo4j                      │
│    ❌ Invalid nodes → Reject & log error                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 7. Create Nodes in Neo4j (Schema-Compliant)                          │
│    CREATE (n:SecurityBehavior {content: "2FA"})                      │
│    CREATE (r:Risk {content: "Account Takeover"})                     │
│    CREATE (n)-[:INTRODUCES_RISK]->(r)                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔀 **Comparison: Where They Differ**

| Step | Auto-Discovery | Custom Schema |
|------|----------------|---------------|
| **Schema Definition** | None (emergent) | Predefined in Parse |
| **LLM Guidance** | Soft (cached patterns as hints) | Strict (enforced node/rel types) |
| **Validation** | None | Pre-creation validation |
| **Evolution** | Auto-learns from usage | Manual schema updates |
| **Error Handling** | Accepts any node type | Rejects invalid types |
| **Cache Update** | After each memory batch | Not needed (schema is static) |

---

## 🧩 **Data Flow: Where Schema is Used**

### Auto-Discovery Path (Current)
```
Memory Creation → Neo4j Storage → Pattern Discovery → Parse Cache (ActiveNodeRel)
                                                           ↓
                                                    LLM Context (next request)
```

### Custom Schema Path (Desired)
```
Parse Schema (UserGraphSchema) → LLM Prompt → Validation → Neo4j Storage
        ↑                                          ↓
   Developer                                 Reject if invalid
```

---

## 🛠️ **Code Locations**

### Auto-Discovery (Active ✅)
```
memory/memory_graph.py:
  - Line 2931: _index_memories_and_process (main entry point)
  - Line 3552: _discover_neo4j_patterns_for_cache (pattern discovery)
  - Line 3499: Cache update trigger

services/active_node_rel_service.py:
  - Line 35: get_cached_schema (fetch cached patterns)
  - Line 147: update_cached_schema (store patterns)

services/auth_utils.py:
  - Line 2710: _get_cached_schema_patterns_direct (fast cache lookup)
```

### Custom Schema (Inactive ❌)
```
routers/v1/schema_routes_v1.py:
  - Line 50: POST /v1/schemas (create schema) ✅
  - Line 150: GET /v1/schemas/{schema_id} (fetch schema) ✅

services/schema_service.py:
  - Line 10: SchemaService (CRUD operations) ✅

models/user_schemas.py:
  - Line 1: UserGraphSchema (Pydantic model) ✅

❌ MISSING:
  - routers/v1/document_routes.py: schema_id parameter
  - cloud_plugins/temporal/workflows/document_processing.py: schema_id propagation
  - cloud_plugins/temporal/activities/document_activities.py: schema_id usage
  - core/document_processing/llm_memory_generator.py: schema enforcement
  - memory/memory_graph.py: _index_memories_and_process schema validation
```

---

## 📊 **Example: Your Three Schemas**

### Schema 1: Customer Support & Workflows
```
Nodes: CallSession, Utterance, Workflow, Step, Agent, Customer
Relationships: HAS, CONTAINS, PERFORMS, FOLLOWS

Current (Auto-Discovery):
  - First upload: LLM creates random nodes
  - After 5 uploads: Patterns emerge (CallSession → HAS → Utterance)
  - Cache guides future uploads

Desired (Custom Schema):
  - Define schema upfront with all 6 node types
  - ALL uploads use EXACTLY these types
  - No drift or inconsistency
```

### Schema 2: Security Protocols
```
Nodes: SecurityBehavior, Risk, Impact, Control, Tool
Relationships: INTRODUCES_RISK, CAUSES, MITIGATES

Current (Auto-Discovery):
  - Inconsistent names: "Security Protocol" vs "SecurityBehavior"
  - Missed relationships between Risk → Impact

Desired (Custom Schema):
  - Strict naming: always "SecurityBehavior"
  - Required relationships enforced
```

---

## ✅ **Action Items to Enable Custom Schemas**

1. **Document Upload API** (`routers/v1/document_routes.py`)
   ```python
   @router.post("/upload")
   async def upload_document_v2(
       schema_id: Optional[str] = Form(None)  # ADD THIS
   )
   ```

2. **Workflow Input** (`cloud_plugins/temporal/workflows/document_processing.py`)
   ```python
   @dataclass
   class DocumentProcessingWorkflow:
       schema_id: Optional[str] = None  # ADD THIS
   ```

3. **LLM Generator** (`core/document_processing/llm_memory_generator.py`)
   ```python
   async def generate_with_schema(content, schema_id):
       schema = await fetch_user_schema(schema_id) if schema_id else None
       # Use schema to constrain LLM output
   ```

4. **Memory Indexing** (`memory/memory_graph.py`)
   ```python
   async def _index_memories_and_process(..., schema_id: Optional[str] = None):
       if schema_id:
           schema = await fetch_user_schema(schema_id)
           await _validate_against_schema(graph_data, schema)
   ```

---

**Would you like me to implement any of these changes?** [[memory:7219971]]

