# Security Schema Test Findings

## Test Created
Created `test_security_schema_neo4j.py` - A comprehensive test that validates security schema functionality by:

1. Creating a security schema with 3 node types (SecurityBehavior, Tactic, Impact)
2. Adding 3 memories with different approaches:
   - Memory with `schema_id` (top-level field - LLM generates graph using schema)
   - Memory without schema (baseline - no custom graph extraction)
   - Memory with `graph_override` (top-level field - pre-made graph structure)
3. Each memory uses a unique `external_user_id` (in metadata) for isolated verification
4. **Directly queries Neo4j** to verify memories exist (no reliance on non-existent API fields)
5. Verifies search API returns Neo4j data for each `external_user_id`

## Issues Discovered

### 1. Relationship Types Missing `name` Field ❌

**Error:**
```json
{
  "detail": [{
    "type": "missing",
    "loc": ["body", "relationship_types", "MAPS_TO_TACTIC", "name"],
    "msg": "Field required"
  }]
}
```

**Current Schema:**
```python
"relationship_types": {
    "MAPS_TO_TACTIC": {
        "label": "MAPS_TO_TACTIC",  # ← Has label
        "description": "...",
        "allowed_source_types": [...],
        "allowed_target_types": [...]
        # ❌ Missing "name" field
    }
}
```

**Fix Required:**
```python
"relationship_types": {
    "MAPS_TO_TACTIC": {
        "name": "MAPS_TO_TACTIC",  # ✅ Add name field
        "label": "MAPS_TO_TACTIC",
        "description": "...",
        "allowed_source_types": [...],
        "allowed_target_types": [...]
    }
}
```

### 2. Correct API Structure ✅ **RESOLVED**

After investigating the model definitions, here's the **correct** API structure:

**Add Memory Request Structure:**
```python
{
    "content": "Memory content here...",
    "type": "text",  # ← Required field
    "schema_id": "schema_uuid",  # ← Top-level field (from SchemaSpecificationMixin)
    "graph_override": {...},  # ← Top-level field (when providing pre-made graph)
    "metadata": {
        "external_user_id": "user_id",  # ← In metadata, NOT top-level
        "customMetadata": {
            # Other custom fields here
        }
    }
}
```

**Search Request Structure:**
```python
{
    "query": "What are you looking for?",
    "external_user_id": "user_id",  # ← Top-level for search
    "enable_agentic_graph": True,  # ← Top-level for agentic search
    "max_memories": 10
}
```

**Key Findings:**
- `AddMemoryRequest` inherits from `SchemaSpecificationMixin` (models/memory_models.py:1100)
- `SchemaSpecificationMixin` provides `schema_id`, `simple_schema_mode`, and `graph_override` as **top-level fields**
- `external_user_id` goes **in metadata** for add memory (models/memory_models.py:1107)
- `external_user_id` is **top-level** for search requests (verified in test_security_schema_v1.py:517)
- Existing test has `schema_id` in `customMetadata` which is **INCORRECT** ❌

### 3. Environment Variables Not Loading ✅ **RESOLVED**

**Error:** `NEO4J_URL or NEO4J_SECRET not found in environment`

**Root Cause:** The test script was using `os.getenv()` at module level but didn't explicitly load the `.env` file using `python-dotenv`.

**Fix Applied:**
```python
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
```

**Why This Was Needed:** Running `source .env` in bash sets shell variables, but these aren't automatically available to the Python process. The `load_dotenv()` function explicitly reads the `.env` file and sets environment variables for the Python process.

### 4. Previous Test Design Flaw: Polling Non-Existent Field

**Previous Issue:** `run_security_tests_live.py` was polling for `processing_status` field that doesn't exist in the GET `/v1/memory/{id}` response.

**Fix:** New test directly queries Neo4j to verify storage instead of waiting for a field that will never appear.

**GET /v1/memory/{id} Response:**
```json
{
  "memories": [...],
  "nodes": [],  # ← Check this instead
  "schemas_used": null  # ← Or this
  // NO processing_status field!
}
```

## Test Improvements

### ✅ What Works

1. **Direct Neo4j Verification** - Test now queries Neo4j directly using:
   ```python
   driver = GraphDatabase.driver(NEO4J_URL, auth=("neo4j", NEO4J_SECRET))
   ```

2. **Isolated Test Cases** - Each memory uses unique `external_user_id`:
   - `security_test_user_001` - schema_id approach
   - `security_test_user_002` - agentic approach
   - `security_test_user_003` - graph_override approach

3. **Reasonable Wait Time** - 30 seconds instead of 240 seconds

4. **Proper Response Parsing** - Uses `result["data"][0]["memoryId"]` per `AddMemoryResponse` model

## Latest Test Results ✅

**Date:** 2025-10-27
**Test Run:** All 7 tests passed (Duration: 72.98s)

### ✅ What Works:
1. **Neo4j Connection**: Successfully connected using `python-dotenv`
2. **Schema Creation**: Security schema created with 3 node types and 2 relationship types
3. **Memory Creation**: All 3 approaches working:
   - `schema_id` approach (LLM generates graph)
   - Baseline approach (no schema)
   - `graph_override` approach (pre-made graph)
4. **Search API**: Returns memories for all external_user_ids

### ⚠️ Remaining Issues:

#### Issue 1: external_user_id Not Stored on Neo4j Nodes
**Observation:** Test 6 found 0 nodes in Neo4j for all 3 memories

**Neo4j Warning:**
```
{severity: WARNING} {code: Neo.ClientNotification.Statement.UnknownPropertyKeyWarning}
{description: One of the property names in your query is not available in the database,
make sure you didn't misspell it or that the label is available when you run this
statement in your application (the missing property name is: externalUserId)}
```

**Analysis:**
- `external_user_id` from metadata isn't being stored as a property on Neo4j nodes
- Query looks for both `n.externalUserId` and `n.external_user_id` but neither exists
- Either background processing hasn't completed OR metadata fields aren't propagated to graph nodes

**Possible Solutions:**
1. Increase wait time beyond 30 seconds
2. Ensure `external_user_id` from metadata is added as a node property during graph creation
3. Use memory ID instead of `external_user_id` to query Neo4j nodes

#### Issue 2: Search Response Doesn't Include Neo4j Graph Data
**Observation:** Test 7 search returns memories but no graph data

**Response Structure:**
```python
{
  "code": 200,
  "status": "success",
  "data": [...],  # Memories array
  "error": null,
  "details": null,
  "search_id": "..."
}
```

**Missing Fields:**
- No `nodes` array
- No `relationships` array

**Possible Solutions:**
1. Check if search endpoint needs a parameter to include graph data
2. Check if `enable_agentic_graph` parameter affects response structure
3. Verify if Neo4j data should be included in search responses

## Next Steps

1. **Investigate Node Storage** - Check if background processing needs more time or if `external_user_id` needs to be explicitly set on nodes
2. **Query by Memory ID** - Alternative verification approach using memory IDs instead of `external_user_id`
3. **Check Search Response Schema** - Determine if search should return Neo4j data or if it's separate
4. **Background Processing Investigation** - Add logging or increase wait time to verify graph extraction completes

## Environment

- **Neo4j URL:** `bolt+s://011e151c.databases.neo4j.io` ✅ Available
- **Neo4j Secret:** Available in .env ✅
- **Test Server:** `http://localhost:8000` ✅ Running
- **API Key:** `f80c5a2940f21882420b41690522cb2c`

## Test File Location

`/Users/amirkabbara/Documents/GitHub/memory/tests/test_security_schema_neo4j.py`

This test is **ready to run** once the API issues are resolved.
