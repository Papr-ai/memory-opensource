# Custom Schema Quick Start Guide

## 🎯 Quick Answer to Your Question

**Q: "Will custom schemas work for document ingestion and memory indexing?"**

**A: Not yet, but here's what needs to be done:**

### Current State ❌
- Custom schemas can be **created** via `/v1/schemas`
- BUT `schema_id` is **NOT propagated** to document workflows or memory activities
- All memories currently use the **system default schema**

### What's Needed ✅
1. Add `schema_id` parameter to document upload API
2. Pass `schema_id` through Temporal workflows → activities
3. Update `_index_memories_and_process` to fetch & use custom schema
4. Update LLM generator to use custom schema for Neo4j extraction

---

## 📋 Your Three Schemas

### 1. **Customer Support & Workflows**
Nodes: `CallSession`, `Utterance`, `Workflow`, `Step`, `WorkflowRun`, `StepEvent`, `Gap`, `Agent`, `Customer`

Relationships: `HAS_UTTERANCE`, `HANDLED_BY`, `WITH_CUSTOMER`, `HAS_STEP`, `NEXT`, `HAS_RUN`, `OF_WORKFLOW`, `HAS_STEPACTION`, `FOR_STEP`, `EVIDENCE_FROM`, `HAS_GAP`

### 2. **Security Protocols & Risk**
Nodes: `SecurityBehavior`, `Control`, `VerificationEvent`, `RiskIndicator`, `Impact`, `Tool`

Relationships: `REQUIRES_CONTROL`, `COVERS_BEHAVIOR`, `HAS_VERIFICATION`, `TRIGGERS_RISK`, `MAPPED_TO_BEHAVIOR`, `MAPPED_TO_IMPACT`, `CAN_LEAD_TO_IMPACT`, `USES_TOOL`

### 3. **Combined Schema** (recommended)
Combine both schemas above into one comprehensive schema for call center operations.

---

## 🚀 Testing Your Schemas (Step by Step)

### Step 1: Create Your Schema

```bash
# Run this test to create schemas in Parse Server
cd /Users/shawkatkabbara/Documents/GitHub/memory
poetry run pytest tests/test_custom_schema_document_ingestion.py::test_create_custom_schemas -xvs
```

This will output schema IDs like:
```
✅ Created Workflow Schema: abc123xyz
✅ Created Security Schema: def456uvw
```

### Step 2: Verify Schema in Parse

Check Parse Server dashboard → `UserGraphSchema` class:
- Verify your schemas exist
- Verify `status` = `active`
- Note the `objectId` (this is your `schema_id`)

### Step 3: Test Schema Retrieval

```python
import httpx

async def test_get_schema(api_key: str, schema_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://localhost:8000/v1/schemas/{schema_id}",
            headers={"X-API-Key": api_key}
        )
        
        schema = response.json()
        print(f"Schema: {schema['data']['name']}")
        print(f"Nodes: {list(schema['data']['node_types'].keys())}")
        print(f"Relationships: {list(schema['data']['relationship_types'].keys())}")
```

### Step 4: Test Document Upload (WILL FAIL - Feature Not Implemented)

```python
# This will IGNORE schema_id currently - needs implementation
async def upload_with_schema(api_key: str, schema_id: str):
    async with httpx.AsyncClient() as client:
        with open("tests/call_answering_sop.pdf", "rb") as f:
            files = {"file": ("sop.pdf", f, "application/pdf")}
            
            metadata = {
                "metadata": {
                    "schema_id": schema_id,  # ⬅️ Currently ignored!
                    "customMetadata": {"doc_type": "SOP"}
                }
            }
            
            response = await client.post(
                "http://localhost:8000/v1/document",
                headers={"X-API-Key": api_key},
                files=files,
                data={"metadata": json.dumps(metadata)}
            )
            
            print(f"Upload ID: {response.json()['document_status']['upload_id']}")
```

### Step 5: Verify Neo4j (After Implementation)

```cypher
// Check what node types were created
MATCH (n)
WHERE labels(n) <> ['Memory']
RETURN DISTINCT labels(n), count(*) as count
ORDER BY count DESC

// Expected output (after implementation):
// ['CallSession'], 5
// ['Workflow'], 2
// ['Step'], 15
// ['Agent'], 3
// etc.

// Current output:
// ['Memory'], 50  ← Only Memory nodes (system default)
```

---

## 🔍 Where Schema is Used

### 1. LLM Memory Generation (`generate_llm_optimized_memory_structures`)
**Current**: Uses system schema (Memory, Goal, UseCase)
**Needed**: Fetch `custom_schema` and pass to LLM generator

**File**: `cloud_plugins/temporal/activities/document_activities.py:1684`

### 2. Memory Indexing (`_index_memories_and_process`)
**Current**: Calls `get_memory_graph_schema()` → returns system schema
**Needed**: Check if `schema_id` provided → fetch custom schema → use instead of default

**File**: `memory/memory_graph.py:3005`

### 3. Schema Generation (`idx_generate_graph_schema`)
**Current**: Activity doesn't receive `schema_id` parameter
**Needed**: Add `schema_id` to activity payload → fetch schema → pass to `process_memory_item_async`

**File**: `cloud_plugins/temporal/activities/memory_activities.py:630`

### 4. Batch Processing (`process_batch_memories_from_parse_reference`)
**Current**: No `schema_id` parameter
**Needed**: Add `schema_id` → pass to batch handler → propagate to individual memory processing

**File**: `cloud_plugins/temporal/activities/memory_activities.py:712`

---

## 📝 Implementation Checklist

Use this checklist to track implementation progress:

### Phase 1: Data Flow
- [ ] Add `schema_id` to `MemoryMetadata` model
- [ ] Add `schema_id` to `AddMemoryRequest` model
- [ ] Add `schema_id` to `BatchMemoryRequest` model
- [ ] Update document upload API to accept `schema_id`
- [ ] Update document workflow to pass `schema_id`
- [ ] Update `create_hierarchical_memory_batch` activity signature
- [ ] Update `process_batch_memories_from_parse_reference` signature
- [ ] Update `idx_generate_graph_schema` activity signature

### Phase 2: Schema Fetching
- [ ] Add `get_active_schema_for_user()` to `schema_service.py`
- [ ] Add `_convert_user_schema_to_llm_schema()` to `memory_graph.py`
- [ ] Update `_index_memories_and_process` to accept `schema_id`
- [ ] Update `_index_memories_and_process` to fetch & use custom schema
- [ ] Update `generate_llm_optimized_memory_structures` to accept `schema_id`
- [ ] Update LLM generator to use custom schema

### Phase 3: Testing
- [ ] Create test schemas (run `test_create_custom_schemas`)
- [ ] Test schema retrieval via API
- [ ] Test document upload with `schema_id` (after implementation)
- [ ] Verify Neo4j has custom nodes/relationships
- [ ] Test batch memory with `schema_id`
- [ ] Test Cypher queries with custom schema

---

## 🎓 Key Insights

### How Schema Propagation Works

```
Document Upload
    ↓ schema_id in metadata
DocumentProcessingWorkflow
    ↓ schema_id passed as arg
extract_structured_content_from_provider
    ↓ schema_id stored in extraction metadata
generate_llm_optimized_memory_structures  ← Fetch custom schema here
    ↓ custom_schema passed to LLM
create_hierarchical_memory_batch
    ↓ schema_id passed to each chunk
process_batch_memories_from_parse_reference
    ↓ schema_id in batch request
_index_memories_and_process  ← Fetch custom schema here
    ↓ custom_schema used for Neo4j generation
Neo4j Graph with Custom Nodes & Relationships ✅
```

### Schema Priority

1. **Explicit `schema_id`**: If provided, use that specific schema
2. **Active workspace schema**: If no explicit ID, use active schema for workspace
3. **System default**: If no custom schema, use system Memory/Goal/UseCase schema

---

## 📚 Full Documentation

See `/Users/shawkatkabbara/Documents/GitHub/memory/docs/CUSTOM_SCHEMA_IMPLEMENTATION_GUIDE.md` for:
- Complete implementation details
- Code examples for each phase
- Migration strategy
- Usage examples
- Cypher query examples

---

## 🤝 Getting Help

If you implement this feature and encounter issues:

1. **Check Logs**: Look for `🎯 CUSTOM SCHEMA:` log lines
2. **Verify Schema Active**: Ensure schema status is `ACTIVE`
3. **Check Temporal UI**: Verify `schema_id` is in workflow input
4. **Query Neo4j**: Verify custom node labels appear
5. **Test with Simple Schema**: Start with 2-3 nodes before complex schemas


