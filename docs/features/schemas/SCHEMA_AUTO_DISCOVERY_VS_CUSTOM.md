# Schema Discovery: Auto-Discovery vs Custom Schemas

## 🎯 **Two Different Systems**

You've identified an important distinction! We currently have **TWO separate schema systems**:

---

## 1️⃣ **Auto-Discovery System (Currently Active)** ✅

This is what **actually runs** in `_index_memories_and_process`:

### How It Works

#### Step 1: **Initial Memory Creation** (lines 2931-3500 in `memory_graph.py`)
- When a memory is indexed, the LLM generates nodes and relationships
- These are created in Neo4j using **whatever labels the LLM decides**
- No predefined schema is enforced

#### Step 2: **Pattern Discovery After Creation** (lines 3552-3605)
```python
async def _discover_neo4j_patterns_for_cache(neo_session, user_id, workspace_id):
    """
    Query Neo4j to find all EXISTING node-relationship-node patterns.
    Returns top 50 patterns ordered by usage count.
    """
    cypher_query = """
    MATCH (source)-[rel]->(target)
    WHERE source.user_id = $user_id AND source.workspace_id = $workspace_id
    WITH labels(source)[0] as source_label, 
         type(rel) as relationship_type, 
         labels(target)[0] as target_label
    RETURN source_label, relationship_type, target_label, count(*) as pattern_count
    ORDER BY pattern_count DESC
    LIMIT 50
    """
```

**Example Output:**
```python
[
    {"source": "Agent", "relationship": "PERFORMS", "target": "Step", "count": 45},
    {"source": "CallSession", "relationship": "HAS", "target": "Utterance", "count": 32},
    {"source": "Workflow", "relationship": "CONTAINS", "target": "Step", "count": 28}
]
```

#### Step 3: **Cache Update in Parse** (`active_node_rel_service.py` lines 147-242)
```python
async def update_cached_schema(user_id, workspace_id, neo4j_patterns):
    """
    Store discovered patterns in Parse 'ActiveNodeRel' class.
    This cache is used for:
    1. LLM context in future memory generation (shows the LLM what patterns exist)
    2. Agentic search (knows which nodes/rels to query)
    3. Graph visualization hints
    """
    parse_data = {
        "user": {"__type": "Pointer", "className": "_User", "objectId": user_id},
        "workspace": {"__type": "Pointer", "className": "WorkSpace", "objectId": workspace_id},
        "activePatterns": json.dumps(sorted_patterns)  # Top 50 patterns
    }
    # Update or create in Parse ActiveNodeRel class
```

#### Step 4: **Using Cached Patterns in Future Requests** (`auth_utils.py` lines 2710-2869)
When a new memory is added:
1. **Fast cache lookup** runs in parallel with authentication
2. Cached patterns are injected into LLM context
3. LLM is **guided** to use existing node/rel types (but not strictly enforced)

---

## 2️⃣ **Custom Schema System (NOT Currently Used)** ❌

This is what you found in `/v1/schemas` API:

### What Exists
- ✅ API to **create** custom schemas (`POST /v1/schemas`)
- ✅ API to **retrieve** schemas (`GET /v1/schemas/{schema_id}`)
- ✅ Parse storage (`UserGraphSchema` class)
- ✅ Schema models with:
  - `node_types`: Define custom node labels + properties
  - `relationship_types`: Define custom relationship types + constraints

### What's Missing
- ❌ `schema_id` is **NOT passed** to document upload
- ❌ `schema_id` is **NOT passed** to Temporal workflows
- ❌ `schema_id` is **NOT passed** to LLM generator
- ❌ `schema_id` is **NOT passed** to `_index_memories_and_process`
- ❌ **No enforcement** of custom schema node/rel types

---

## 🔍 **Key Differences**

| Aspect | Auto-Discovery | Custom Schemas |
|--------|----------------|----------------|
| **Definition** | Learns from Neo4j | Predefined by developer |
| **Enforcement** | Soft guidance (LLM context) | Should be strict (not implemented) |
| **Timing** | After creation (reactive) | Before creation (proactive) |
| **Usage** | Active ✅ | Not used ❌ |
| **Storage** | Parse `ActiveNodeRel` | Parse `UserGraphSchema` |
| **Scope** | Per user/workspace | Per schema_id |
| **Evolution** | Auto-updates with usage | Requires manual schema updates |

---

## 📊 **Example: How Your Three Schemas Would Work**

### Current Behavior (Auto-Discovery):
```python
# Upload document
POST /v1/documents/upload
Body: {"file": "two-factor-authentication.pdf"}
# No schema_id provided!

# What happens:
1. LLM generates memories with RANDOM node types:
   - "Security Protocol", "Authentication Method", "Risk Factor"
2. After indexing, patterns are discovered:
   - ("Security Protocol", "MITIGATES", "Risk Factor") count=5
3. Patterns cached in Parse ActiveNodeRel
4. Next document uses these patterns as HINTS (but not enforced)
```

### Desired Behavior (Custom Schemas):
```python
# Create custom schema first
POST /v1/schemas
Body: {
  "name": "Security & Workflows",
  "node_types": {
    "SecurityBehavior": {...},
    "Risk": {...},
    "Impact": {...},
    "Workflow": {...},
    "Step": {...}
  },
  "relationship_types": {
    "INTRODUCES_RISK": {"source": "SecurityBehavior", "target": "Risk"},
    "CAUSES": {"source": "Risk", "target": "Impact"},
    "CONTAINS": {"source": "Workflow", "target": "Step"}
  }
}
# Returns: {"schema_id": "abc123"}

# Upload document WITH schema_id
POST /v1/documents/upload
Body: {
  "file": "two-factor-authentication.pdf",
  "schema_id": "abc123"  # ❌ Not currently supported!
}

# What SHOULD happen (not implemented):
1. Fetch schema "abc123" from Parse
2. Pass schema to LLM with STRICT instructions
3. LLM MUST use only: SecurityBehavior, Risk, Impact, Workflow, Step
4. Validate generated graph against schema (reject invalid nodes)
5. Index with schema constraints enforced
```

---

## 🛠️ **Implementation Plan for Custom Schemas**

### Phase 1: Schema Propagation
```python
# 1. Add schema_id to document upload
POST /v1/documents/upload
Body: {"file": "...", "schema_id": "abc123", "metadata": {...}}

# 2. Pass through workflow
DocumentProcessingWorkflow.run(
    upload_id=...,
    schema_id="abc123"  # NEW
)

# 3. Pass to LLM activity
generate_llm_optimized_memory_structures(
    content=...,
    schema_id="abc123"  # NEW
)
```

### Phase 2: Schema Enforcement in LLM
```python
# In llm_memory_generator.py
async def generate_with_schema(content, schema_id):
    # Fetch schema from Parse
    schema = await fetch_user_schema(schema_id)
    
    # Build strict LLM prompt
    allowed_nodes = schema.node_types.keys()
    allowed_rels = schema.relationship_types.keys()
    
    prompt = f"""
    STRICT SCHEMA ENFORCEMENT:
    - ONLY use these node types: {allowed_nodes}
    - ONLY use these relationships: {allowed_rels}
    - ANY other nodes/relationships are INVALID and will be REJECTED
    
    Content to analyze:
    {content}
    """
```

### Phase 3: Schema Validation
```python
# In memory_graph.py
async def _validate_against_schema(graph_data, schema_id):
    schema = await fetch_user_schema(schema_id)
    
    for node in graph_data["nodes"]:
        if node["label"] not in schema.node_types:
            raise SchemaValidationError(f"Invalid node type: {node['label']}")
    
    for rel in graph_data["relationships"]:
        if rel["type"] not in schema.relationship_types:
            raise SchemaValidationError(f"Invalid relationship: {rel['type']}")
```

---

## 🧪 **Testing Your Three Schemas**

### Step 1: Create the schemas
```bash
# Create each schema
poetry run python tests/test_custom_schema_document_ingestion.py::test_create_all_three_schemas
```

### Step 2: Update document upload to accept schema_id
```python
# In routers/v1/document_routes.py
@router.post("/upload")
async def upload_document_v2(
    file: UploadFile,
    schema_id: Optional[str] = Form(None),  # NEW
    ...
):
    # Pass schema_id to workflow
    await client.execute_workflow(
        DocumentProcessingWorkflow.run,
        args=[...],
        task_queue="document-processing",
        id=f"doc-{upload_id}",
        # Store schema_id in workflow input
    )
```

### Step 3: Test end-to-end
```bash
# Upload with custom schema
curl -X POST "http://localhost:8000/v1/documents/upload" \
  -H "X-API-Key: your-key" \
  -F "file=@two-factor-authentication.pdf" \
  -F "schema_id=abc123"

# Verify nodes match schema
# Query Neo4j to check only SecurityBehavior, Risk, Impact, Workflow, Step nodes exist
```

---

## 📝 **Summary**

**Current State:**
- ✅ **Auto-discovery works** - learns patterns from existing Neo4j graph
- ✅ **Caching works** - stores patterns in Parse for LLM guidance
- ❌ **Custom schemas don't work** - no propagation or enforcement

**To Make Custom Schemas Work:**
1. Add `schema_id` parameter to document upload API
2. Pass `schema_id` through workflows → activities  
3. Fetch schema in LLM generator and `_index_memories_and_process`
4. Enforce schema constraints (strict validation)
5. Test with your three domain-specific schemas

**Hybrid Approach (Recommended):**
- Use **custom schemas** for structured domains (security, workflows, support)
- Use **auto-discovery** for exploratory/unstructured data
- Allow `schema_id=null` to default to auto-discovery mode

---

## 🎯 **Next Steps**

Would you like me to:
1. ✅ **Implement** schema propagation (add `schema_id` parameter throughout the pipeline)?
2. ✅ **Create** integration test for your three schemas?
3. ✅ **Add** schema validation/enforcement in LLM generator?
4. ✅ **Document** best practices for hybrid auto-discovery + custom schemas?

Let me know which path you want to take! [[memory:7219971]]

