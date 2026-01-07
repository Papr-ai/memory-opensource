# Schema-Based Document Processing

This guide explains how to use custom schemas with PAPR Memory's hierarchical document processing pipeline to create structured knowledge graphs from your documents.

## Overview

When you upload a document to PAPR Memory, it goes through a multi-stage pipeline that:

1. **Extracts content** using AI-powered document processors (TensorLake, Reducto, or Gemini)
2. **Chunks intelligently** using hierarchical semantic chunking
3. **Generates memories** with LLM-enhanced metadata
4. **Creates knowledge graphs** using your custom schema (if provided)

By providing a `schema_id`, you can control how the extracted content is structured into nodes and relationships in your knowledge graph.

## Quick Start

### Upload with Schema ID

```bash
curl -X POST "https://api.papr.ai/v1/document" \
  -H "X-API-Key: your-api-key" \
  -F "file=@your-document.pdf" \
  -F "schema_id=your-schema-id" \
  -F "hierarchical_enabled=true"
```

### Python SDK

```python
from papr_memory import PaprMemory

client = PaprMemory(api_key="your-api-key")

response = client.upload_document(
    file_path="your-document.pdf",
    schema_id="your-schema-id",
    hierarchical_enabled=True
)

print(f"Upload ID: {response.upload_id}")
print(f"Status: {response.status}")
```

---

## How It Works

### Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Document Upload (POST /v1/document)                  │
│                                                                         │
│  Parameters:                                                            │
│  • file: PDF, DOCX, etc.                                               │
│  • schema_id: Your custom schema ID                                    │
│  • hierarchical_enabled: true (recommended)                            │
│  • simple_schema_mode: false (default)                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              Stage 1: Document Processing (Provider Extraction)          │
│                                                                         │
│  • TensorLake/Reducto/Gemini extracts text, tables, images              │
│  • Provider-specific structured data is preserved                       │
│  • Creates a Post in Parse Server with provider results                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              Stage 2: Hierarchical Chunking                              │
│                                                                         │
│  • Analyzes document structure (sections, headers, paragraphs)         │
│  • Groups related content into semantic chunks (1-2 pages each)        │
│  • Preserves tables and images as separate chunks                      │
│  • Adds context (400 chars before/after) for better retrieval          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              Stage 3: LLM-Enhanced Metadata Generation                   │
│                                                                         │
│  For each chunk:                                                        │
│  • Generates descriptive titles                                         │
│  • Extracts entities and topics                                        │
│  • Creates search keywords and query patterns                          │
│  • Validates chunk coherence                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              Stage 4: Batch Memory Indexing                              │
│                                                                         │
│  For each memory:                                                       │
│  • Creates Qdrant vector embeddings                                    │
│  • Stores in Parse Server                                              │
│  • ─────────────────────────────────────────────────────────          │
│  │  schema_id is used HERE to generate graph structure  │              │
│  • ─────────────────────────────────────────────────────────          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              Stage 5: Graph Schema Generation (with YOUR schema)         │
│                                                                         │
│  When schema_id is provided:                                            │
│  1. Fetches your schema definition from Parse Server                   │
│  2. Uses schema to constrain LLM graph generation                      │
│  3. Creates nodes matching your defined labels                         │
│  4. Creates relationships matching your defined types                  │
│  5. Applies property overrides if specified                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              Stage 6: Neo4j Relationship Building                        │
│                                                                         │
│  • Creates nodes in Neo4j with your schema's labels                    │
│  • Creates relationships with your schema's types                      │
│  • Links memories to document Post                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Schema Specification Parameters

### API Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `schema_id` | string | ID of your custom schema to enforce during graph generation |
| `simple_schema_mode` | boolean | Use simplified schema mode (default: false) |
| `graph_override` | JSON | Manual graph structure override (advanced) |
| `property_overrides` | JSON | Override node properties with custom values |

### Example: Financial Document Processing

```bash
curl -X POST "https://api.papr.ai/v1/document" \
  -H "X-API-Key: your-api-key" \
  -F "file=@quarterly-report.pdf" \
  -F "schema_id=financial-schema-v1" \
  -F "hierarchical_enabled=true" \
  -F "metadata={\"domain\": \"financial\"}"
```

With a financial schema, your document might generate:

```
(Document)-[:CONTAINS]->(Section:FinancialSummary)
(Section)-[:HAS_METRIC]->(Metric:Revenue {value: "10M", period: "Q4 2024"})
(Metric)-[:COMPARED_TO]->(Metric:Revenue {value: "8M", period: "Q3 2024"})
```

---

## Hierarchical Chunking Details

### What It Does

The hierarchical chunker preserves document structure while creating optimal chunks for:
- **Vector search** (embedding-based retrieval)
- **Graph queries** (relationship-based navigation)
- **Contextual retrieval** (surrounding context for better understanding)

### Chunking Strategy

```python
# Default chunking configuration
config = {
    "strategy": "HIERARCHICAL",
    "max_chunk_size": 6000,      # ~1-2 pages
    "min_chunk_size": 1000,      # Avoid tiny fragments
    "overlap_size": 200,        # Context overlap
    "preserve_tables": True,    # Keep tables as separate chunks
    "preserve_images": True,    # Keep images as separate chunks
    "semantic_threshold": 0.75  # Group similar content
}
```

### Content Types Handled

| Content Type | Handling |
|-------------|----------|
| **Text** | Grouped by section, split at semantic boundaries |
| **Tables** | Preserved as single chunks with surrounding context |
| **Images** | Extracted, uploaded to storage, linked with description |
| **Code** | Preserved with syntax highlighting metadata |
| **Formulas** | Preserved with LaTeX/math context |

---

## Schema Flow Through the Pipeline

### 1. Upload Request

The `schema_id` is captured from your API request:

```python
# document_routes_v2.py
schema_specification = {
    "schema_id": schema_id,              # Your schema ID
    "simple_schema_mode": simple_schema_mode,
    "graph_override": graph_override_obj,
    "property_overrides": property_overrides_obj
}
```

### 2. Temporal Workflow

The schema specification is passed through the durable Temporal workflow:

```python
# DocumentProcessingWorkflow
await workflow.execute_activity(
    "store_batch_memories_in_parse_for_processing",
    args=[
        memory_requests,
        organization_id,
        namespace_id,
        user_id,
        workspace_id,
        document_post_id,
        schema_specification,  # Schema flows here
        file_url
    ]
)
```

### 3. Batch Memory Processing

The batch workflow extracts and uses your schema:

```python
# ProcessBatchMemoryFromPostWorkflow
schema_id = schema_specification.get("schema_id")
simple_schema_mode = schema_specification.get("simple_schema_mode", False)

# Schema is injected into each memory's metadata
for mem_dict in memory_requests:
    mem_dict["metadata"]["customMetadata"]["schema_id"] = schema_id
```

### 4. Graph Schema Generation

The actual schema enforcement happens during graph generation:

```python
# idx_generate_graph_schema activity
result = await memory_graph.process_memory_item_async(
    memory_dict=memory_dict,
    schema_id=schema_id,           # Your schema enforced here
    simple_schema_mode=simple_schema_mode,
    graph_override=graph_override,
    property_overrides=property_overrides
)
```

### 5. LLM Schema Enforcement

The ChatGPT completion handler fetches and applies your schema:

```python
# chat_gpt_completion.py - generate_memory_graph_schema_async
if schema_ids and len(schema_ids) > 0:
    # Fetch your schema definition
    user_schemas = await schema_service.get_schemas_by_ids(
        [selected_schema_id],
        user_id,
        workspace_id,
        organization_id,
        namespace_id
    )
    
    # Schema constrains the LLM output to your defined:
    # - Node labels (e.g., Person, Company, Document)
    # - Relationship types (e.g., WORKS_FOR, AUTHORED, MENTIONS)
    # - Property structures
```

---

## Creating Custom Schemas

### Schema Structure

```json
{
  "name": "Financial Document Schema",
  "description": "Schema for processing financial reports",
  "nodes": [
    {
      "label": "Document",
      "properties": ["title", "date", "type"]
    },
    {
      "label": "Company",
      "properties": ["name", "ticker", "sector"]
    },
    {
      "label": "Metric",
      "properties": ["name", "value", "period", "currency"]
    }
  ],
  "relationships": [
    {
      "type": "MENTIONS",
      "from": "Document",
      "to": "Company"
    },
    {
      "type": "HAS_METRIC",
      "from": "Document",
      "to": "Metric"
    },
    {
      "type": "COMPARED_TO",
      "from": "Metric",
      "to": "Metric"
    }
  ]
}
```

### Register Schema via API

```bash
curl -X POST "https://api.papr.ai/v1/schema" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Financial Document Schema",
    "schema": { ... }
  }'
```

---

## Advanced Options

### Property Overrides

Apply custom property values to generated nodes:

```json
{
  "property_overrides": [
    {
      "match_conditions": {
        "label": "Document"
      },
      "overrides": {
        "source": "quarterly_reports",
        "processed_by": "papr_pipeline_v2"
      }
    }
  ]
}
```

### Graph Override (Manual Mode)

For complete control, provide explicit graph structure:

```json
{
  "graph_override": {
    "nodes": [
      {"id": "doc1", "label": "Document", "properties": {"title": "Q4 Report"}},
      {"id": "company1", "label": "Company", "properties": {"name": "Acme Corp"}}
    ],
    "relationships": [
      {"source_node_id": "doc1", "target_node_id": "company1", "relationship_type": "ABOUT"}
    ]
  }
}
```

---

## Status Tracking

### Check Processing Status

```bash
curl -X GET "https://api.papr.ai/v1/document/status/{upload_id}" \
  -H "X-API-Key: your-api-key"
```

### Status Response

```json
{
  "upload_id": "abc123",
  "status": "completed",
  "progress": 1.0,
  "total_pages": 25,
  "page_id": "post_xyz789",
  "workflow_type": "temporal"
}
```

### Pipeline Stages

| Status | Description |
|--------|-------------|
| `processing` | Document being extracted by provider |
| `analyzing_structure` | Hierarchical chunking in progress |
| `creating_memories` | LLM generating memory structures |
| `indexing_memories` | Batch indexing with schema enforcement |
| `storing_document` | Finalizing Parse Server records |
| `completed` | All stages complete |
| `failed` | Error occurred (check error field) |

---

## Best Practices

### 1. Schema Design

- Keep schemas focused on your domain
- Use consistent naming conventions for labels
- Define all expected relationship types
- Document property constraints

### 2. Document Preparation

- Use clear section headers
- Include tables in standard formats
- Ensure images have descriptive context

### 3. Processing Configuration

- Enable `hierarchical_enabled: true` for complex documents
- Use `simple_schema_mode: true` for basic structure
- Set appropriate `metadata.domain` for better LLM understanding

### 4. Monitoring

- Use webhooks for completion notifications
- Check status endpoint for progress
- Monitor `page_id` for the created Post

---

## Troubleshooting

### Schema Not Applied

1. Verify `schema_id` exists and you have access
2. Check organization/namespace permissions
3. Review logs for "SCHEMA ENFORCEMENT" messages

### Chunking Issues

1. Large documents may timeout - check workflow status
2. Tables not preserved? Ensure `preserve_tables: true`
3. Check `chunking_validation` in memory metadata

### Graph Not Generated

1. Verify schema has matching node labels
2. Check if content matches schema constraints
3. Review `schema_metrics` in response

---

## Related Documentation

- [Schema Management Guide](./schema-management.md)
- [Batch Memory Processing](./batch-memory-processing.md)
- [Knowledge Graph Queries](./knowledge-graph-queries.md)
- [Document Processing Providers](./document-providers.md)

