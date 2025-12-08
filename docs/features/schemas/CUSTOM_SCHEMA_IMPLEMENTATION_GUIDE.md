# Custom Schema Implementation Guide

## Overview

This guide explains how to enable **custom schema support** for document ingestion and memory processing. Currently, all memories use the **system default schema**. This implementation will allow developers to specify a `schema_id` that defines custom Neo4j node types and relationships for their domain.

---

## Current State (Before Implementation)

### ✅ What Works
- Creating custom schemas via `/v1/schemas` API
- Storing schemas in Parse Server (`UserGraphSchema` class)
- Retrieving schemas via `/v1/schemas/{schema_id}`

### ❌ What's Missing
- `schema_id` is **not** passed through document upload → workflows → activities
- LLM memory generation uses **system default schema only**
- `_index_memories_and_process` doesn't fetch user's custom schema
- No way to specify schema for batch memory operations

---

## Implementation Plan

### Phase 1: Add `schema_id` to Data Flow

#### 1.1 Update Models

**File: `models/shared_types.py`**
```python
class MemoryMetadata(BaseModel):
    # ... existing fields ...
    schema_id: Optional[str] = Field(None, description="Custom schema ID for Neo4j graph generation")
```

**File: `models/memory_models.py`**
```python
class AddMemoryRequest(BaseModel):
    # ... existing fields ...
    schema_id: Optional[str] = Field(None, description="Custom schema ID to use for this memory")

class BatchMemoryRequest(BaseModel):
    # ... existing fields ...
    schema_id: Optional[str] = Field(None, description="Custom schema ID for entire batch")
```

#### 1.2 Update Document Upload API

**File: `routers/v1/document_routes.py`**
```python
@router.post("", response_model=DocumentUploadResponse)
async def upload_document_v1(
    # ... existing parameters ...
    schema_id: Optional[str] = Query(None, description="Custom schema ID for memory indexing")
):
    # Extract schema_id from metadata or query param
    if upload_doc_request and upload_doc_request.metadata:
        schema_id = upload_doc_request.metadata.schema_id or schema_id
    
    # Pass schema_id to memory_request
    add_memory_request_document = AddMemoryRequest(
        content="",
        type=upload_document_request.type,
        metadata=upload_document_request.metadata,
        schema_id=schema_id  # ⬅️ NEW
    )
```

#### 1.3 Propagate Through Temporal Workflows

**File: `cloud_plugins/temporal/workflows/document_processing.py`**
```python
@workflow.defn
class DocumentProcessingWorkflow:
    @workflow.run
    async def run(self, input: DocumentWorkflowInput) -> DocumentWorkflowOutput:
        # ... existing code ...
        
        # Extract schema_id from metadata
        schema_id = (metadata or {}).get("schema_id")
        
        # Pass to create_hierarchical_memory_batch
        batch_result = await workflow.execute_activity(
            "create_hierarchical_memory_batch",
            args=[
                memory_requests,
                user_id,
                organization_id,
                namespace_id,
                workspace_id,
                50,  # batch_size
                document_post_id,  # post_id
                schema_id  # ⬅️ NEW
            ],
            # ... timeouts ...
        )
```

#### 1.4 Update Activities

**File: `cloud_plugins/temporal/activities/document_activities.py`**
```python
@activity.defn
async def create_hierarchical_memory_batch(
    memory_requests: List[Dict[str, Any]],
    user_id: str,
    organization_id: str,
    namespace_id: str,
    workspace_id: str,
    batch_size: int = 20,
    post_id: Optional[str] = None,
    schema_id: Optional[str] = None  # ⬅️ NEW
) -> Dict[str, Any]:
    # Pass schema_id to process_batch_memories_from_parse_reference
    batch_result = await process_batch_memories_from_parse_reference(
        post_id=chunk_post_id,
        organization_id=organization_id,
        namespace_id=namespace_id,
        user_id=user_id,
        workspace_id=workspace_id,
        schema_id=schema_id  # ⬅️ NEW
    )
```

**File: `cloud_plugins/temporal/activities/memory_activities.py`**
```python
@activity.defn(name="process_batch_memories_from_parse_reference")
async def process_batch_memories_from_parse_reference(
    post_id: str,
    organization_id: str,
    namespace_id: str,
    user_id: str,
    workspace_id: Optional[str] = None,
    schema_id: Optional[str] = None  # ⬅️ NEW
) -> Dict[str, Any]:
    # Add schema_id to batch request
    batch_request = BatchMemoryRequest(
        memories=add_memory_requests,
        organization_id=organization_id,
        namespace_id=namespace_id,
        user_id=user_id,
        external_user_id=user_id,
        workspace_id=workspace_id,
        batch_size=min(50, len(add_memory_requests) or 1),
        schema_id=schema_id  # ⬅️ NEW
    )
```

---

### Phase 2: Fetch & Apply Custom Schema

#### 2.1 Add Schema Fetching Utility

**File: `services/schema_service.py`**
```python
async def get_active_schema_for_user(
    user_id: str,
    workspace_id: Optional[str] = None,
    schema_id: Optional[str] = None
) -> Optional[UserGraphSchema]:
    """
    Fetch the active schema for a user.
    
    Priority:
    1. If schema_id provided, fetch that specific schema
    2. Otherwise, fetch the active schema for user/workspace
    3. If no custom schema, return None (use system default)
    """
    schema_service = get_schema_service()
    
    if schema_id:
        # Fetch specific schema
        response = await schema_service.get_schema(schema_id, user_id)
        if response.success:
            return response.data
    else:
        # Fetch active schemas for user/workspace
        response = await schema_service.list_schemas(user_id, workspace_id)
        if response.success and response.data:
            # Find first ACTIVE schema
            for schema in response.data:
                if schema.status == SchemaStatus.ACTIVE:
                    return schema
    
    return None  # Use system default
```

#### 2.2 Update `_index_memories_and_process`

**File: `memory/memory_graph.py`**
```python
async def _index_memories_and_process(
    self, 
    neo_session: AsyncSession, 
    session_token: str, 
    memory_dict: dict, 
    # ... existing params ...
    schema_id: Optional[str] = None  # ⬅️ NEW
) -> ProcessMemoryResponse:
    
    # ... existing code ...
    
    # STEP: Fetch custom schema if provided
    custom_schema = None
    if schema_id:
        from services.schema_service import get_active_schema_for_user
        custom_schema = await get_active_schema_for_user(
            user_id=user_id,
            workspace_id=workspace_id,
            schema_id=schema_id
        )
    
    # Get memory graph schema (custom or default)
    if custom_schema:
        logger.info(f"🎯 CUSTOM SCHEMA: Using schema '{custom_schema.name}' ({custom_schema.id})")
        memory_graph_schema = self._convert_user_schema_to_llm_schema(custom_schema)
    else:
        logger.info(f"🤖 DEFAULT SCHEMA: Using system schema")
        memory_graph_schema = self.get_memory_graph_schema()
    
    # ... rest of processing uses memory_graph_schema ...
```

#### 2.3 Convert UserGraphSchema to LLM Format

**File: `memory/memory_graph.py`**
```python
def _convert_user_schema_to_llm_schema(self, user_schema: UserGraphSchema) -> Dict[str, Any]:
    """
    Convert UserGraphSchema to the format expected by LLM generation.
    
    Returns:
        {
            "nodes": ["Memory", "CallSession", "Workflow", "Step", ...],
            "relationships": ["HAS_UTTERANCE", "HAS_STEP", "NEXT", ...]
        }
    """
    # Extract node labels
    node_labels = ["Memory"]  # Always include Memory
    if user_schema.node_types:
        node_labels.extend(list(user_schema.node_types.keys()))
    
    # Extract relationship types
    relationship_types = []
    if user_schema.relationship_types:
        relationship_types.extend(list(user_schema.relationship_types.keys()))
    
    logger.info(f"🎯 CUSTOM SCHEMA CONVERSION: {len(node_labels)} nodes, {len(relationship_types)} relationships")
    logger.info(f"  Nodes: {node_labels}")
    logger.info(f"  Relationships: {relationship_types}")
    
    return {
        "nodes": node_labels,
        "relationships": relationship_types
    }
```

#### 2.4 Update LLM Memory Generation

**File: `cloud_plugins/temporal/activities/document_activities.py`**
```python
@activity.defn
async def generate_llm_optimized_memory_structures(
    content_elements: List[Dict[str, Any]],
    domain: Optional[str],
    base_metadata: MemoryMetadata,
    organization_id: str,
    namespace_id: str,
    use_llm: bool = True,
    post_id: Optional[str] = None,
    extraction_stored: bool = False,
    schema_id: Optional[str] = None  # ⬅️ NEW
) -> Dict[str, Any]:
    
    # Fetch custom schema if provided
    custom_schema = None
    if schema_id:
        from services.schema_service import get_active_schema_for_user
        custom_schema = await get_active_schema_for_user(
            user_id=base_metadata.user_id,
            workspace_id=base_metadata.workspace_id,
            schema_id=schema_id
        )
    
    # Pass custom schema to LLM generator
    if use_llm:
        memory_requests = await generate_optimized_memory_structures(
            content_elements=elements,
            domain=domain,
            base_metadata=metadata,
            custom_schema=custom_schema  # ⬅️ NEW
        )
```

**File: `core/document_processing/llm_memory_generator.py`**
```python
async def generate_optimized_memory_structures(
    content_elements: List[ContentElement],
    domain: Optional[str] = None,
    base_metadata: Optional[MemoryMetadata] = None,
    custom_schema: Optional[UserGraphSchema] = None  # ⬅️ NEW
) -> List[AddMemoryRequest]:
    
    generator = LLMMemoryStructureGenerator()
    
    # Override system schema with custom schema if provided
    if custom_schema:
        logger.info(f"🎯 CUSTOM SCHEMA: Generating memories using '{custom_schema.name}'")
        generator.set_custom_schema(custom_schema)
    
    return await generator.generate_batch_memory_structures(
        content_elements=content_elements,
        domain_context=DomainContext(domain=domain) if domain else None,
        base_metadata=base_metadata
    )
```

---

### Phase 3: Testing & Validation

#### 3.1 Unit Tests

**File: `tests/test_custom_schema_memory_processing.py`**
```python
@pytest.mark.asyncio
async def test_memory_processing_with_custom_schema():
    """Test that custom schema is used in memory processing"""
    
    # Create custom schema
    schema = UserGraphSchema(
        name="Test Schema",
        node_types={"CustomNode": {...}},
        relationship_types={"CUSTOM_REL": {...}}
    )
    schema_id = await create_schema(schema)
    
    # Create memory with schema_id
    memory_request = AddMemoryRequest(
        content="Test content",
        schema_id=schema_id
    )
    
    # Process memory
    result = await process_memory(memory_request)
    
    # Verify custom nodes were created in Neo4j
    assert "CustomNode" in result.graph_nodes
    assert "CUSTOM_REL" in result.graph_relationships
```

#### 3.2 Integration Tests

Use the test script created above: `tests/test_custom_schema_document_ingestion.py`

```bash
# Run integration test
poetry run pytest tests/test_custom_schema_document_ingestion.py::test_create_custom_schemas -xvs
```

---

## Usage Examples

### Example 1: Document Upload with Custom Schema

```python
import httpx

async def upload_document_with_schema(api_key: str, schema_id: str, file_path: str):
    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "application/pdf")}
            
            metadata = {
                "metadata": {
                    "schema_id": schema_id,  # ⬅️ Specify custom schema
                    "customMetadata": {
                        "document_type": "SOP"
                    }
                }
            }
            
            response = await client.post(
                "http://localhost:8000/v1/document",
                headers={"X-API-Key": api_key},
                files=files,
                data={"metadata": json.dumps(metadata)}
            )
            
            return response.json()
```

### Example 2: Batch Memory with Custom Schema

```python
async def add_batch_with_schema(api_key: str, schema_id: str, memories: List[str]):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/memory/batch",
            headers={"X-API-Key": api_key},
            json={
                "memories": [{"content": m} for m in memories],
                "schema_id": schema_id  # ⬅️ Apply schema to entire batch
            }
        )
        
        return response.json()
```

### Example 3: Query with Schema-Specific Nodes

```cypher
// After document is processed with custom schema, query Neo4j:

// Find CallSessions with high risk
MATCH (cs:CallSession)-[:TRIGGERS_RISK]->(ri:RiskIndicator)
WHERE ri.severity >= 4
RETURN cs, ri

// Find workflow gaps
MATCH (wr:WorkflowRun)-[:HAS_GAP]->(g:Gap)
WHERE g.severity >= 3
RETURN wr, g

// Find security behavior violations
MATCH (wf:Workflow)-[:REQUIRES_CONTROL]->(c:Control)
WHERE NOT EXISTS {
  MATCH (cs:CallSession)-[:HAS_RUN]->(wr:WorkflowRun)-[:OF_WORKFLOW]->(wf)
  MATCH (cs)-[:HAS_VERIFICATION]->(ve:VerificationEvent)
  WHERE ve.method = c.name AND ve.status = 'passed'
}
RETURN wf, c
```

---

## Migration Strategy

### Step 1: Schema Creation
1. Create your 3 schemas via `/v1/schemas` API
2. Mark them as `ACTIVE`
3. Note the `schema_id` for each

### Step 2: Test with Single Document
1. Upload a test document with `schema_id`
2. Monitor Temporal workflow logs
3. Verify Neo4j contains custom nodes

### Step 3: Validate Queries
1. Run Cypher queries to verify schema structure
2. Test vector search with custom node types
3. Verify relationships are correct

### Step 4: Production Rollout
1. Update existing document uploads to include `schema_id`
2. Reprocess critical documents with custom schemas
3. Update dashboards/queries to use custom node types

---

## Benefits

✅ **Domain-Specific Knowledge Graphs**: Customer support ops see `CallSession`, `Workflow`, `Step` nodes
✅ **Better LLM Context**: Custom schemas guide LLM to extract domain entities
✅ **Flexible Schema Evolution**: Update schemas without code changes
✅ **Multi-Tenant**: Different customers can have different schemas
✅ **Queryability**: Domain-specific Cypher queries are more intuitive

---

## Next Steps

1. **Implement Phase 1**: Add `schema_id` propagation through data flow
2. **Implement Phase 2**: Fetch & apply custom schemas in memory processing
3. **Run Tests**: Verify with `test_custom_schema_document_ingestion.py`
4. **Create Documentation**: API docs showing schema usage
5. **Production Deploy**: Roll out to first customer with custom schema needs


