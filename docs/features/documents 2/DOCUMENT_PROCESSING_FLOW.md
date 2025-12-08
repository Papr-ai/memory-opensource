# Document Processing Flow

## Overview

This document describes the complete document processing pipeline, from file upload to memory creation, with a focus on the hierarchical chunking and LLM optimization paths.

## Architecture: Simple vs Complex Paths

The system intelligently branches based on document structure analysis:

- **Simple Path**: Documents without complex structures (no tables/images/charts) → Direct Markdown rendering → Size-based chunking → Batch memory creation
- **Complex Path**: Documents with tables, images, charts → Structured element extraction → Hierarchical chunking → LLM optimization → Batch memory creation

## Workflow Steps

### 1. File Upload & Processing (`process_document_with_provider_from_reference`)

**Input**: File reference (URL, name, size)
**Output**: Parse Post with provider JSON stored as file

- Downloads file from Parse Server storage
- Processes with configured provider (Reducto, TensorLake, Gemini, etc.)
- Stores full provider JSON in Parse Post (to avoid Temporal payload size limits)
- Returns minimal payload with `post_id` reference

### 2. Structure Analysis & Content Extraction (`extract_structured_content_from_provider`)

**Input**: `post_id` reference (or direct provider JSON)
**Output**: Decision ("simple" or "complex") + structured elements or markdown chunks

#### 2a. Fetch Provider Results
```python
# Uses reusable helper from services/memory_management.py
post_data = await fetch_post_with_provider_result_async(post_id)
provider_specific = post_data.provider_specific  # Full JSON downloaded & decompressed
provider_name = post_data.provider_name
```

#### 2b. Typed Parse with Provider SDK
```python
# Uses core/document_processing/provider_type_parser.py
from reducto.types.shared.pipeline_response import PipelineResponse
pipeline = PipelineResponse(**provider_specific)  # Type-safe parsing
```

#### 2c. Structure Analysis
```python
analysis = {
    "has_tables": bool,
    "has_images": bool,
    "has_charts": bool,
    "total_pages": int
}
decision = "complex" if (has_tables or has_images or has_charts) else "simple"
```

#### 2d. Content Extraction

**Simple Path**:
```python
markdown = provider_to_markdown(provider_name, provider_specific)
chunks = chunk_text_by_bytes(markdown, 14900)
memory_requests = [AddMemoryRequest(content=chunk, type=DOCUMENT) for chunk in chunks]
return {"decision": "simple", "memory_requests": memory_requests}
```

**Complex Path (Reducto Example)**:
```python
# Navigate Reducto SDK structure
chunks = pipeline.result.parse.result.chunks  # 96 semantic chunks

structured_elements = []
for chunk in chunks:
    for block in chunk.blocks:
        # Extract with typed accessors
        content = block.content
        block_type = str(block.type)  # "Figure", "Text", "Table", etc.
        confidence = block.confidence  # "high", "medium", "low" → mapped to numeric
        
        # Create typed element
        if block_type.lower() in ("table", "table_cell"):
            element = TableElement(content=content, structured_data={...})
        elif block_type.lower() in ("image", "figure"):
            element = ImageElement(content=content, image_url=block.url)
        else:
            element = TextElement(content=content)
        
        structured_elements.append(element)

return {"decision": "complex", "structured_elements": structured_elements}
```

**Real Example Output** (30-page Google research paper):
- 96 semantic chunks
- 359 total elements:
  - 342 text blocks
  - 10 images  
  - 7 tables

### 3. LLM-Optimized Memory Generation (`generate_llm_optimized_memory_structures`)

**Input**: List of `ContentElement` objects (TextElement, TableElement, ImageElement)
**Output**: List of enhanced `AddMemoryRequest` objects

**Only runs for Complex path**

```python
from core.document_processing.llm_memory_generator import generate_optimized_memory_structures

memory_requests = await generate_optimized_memory_structures(
    content_elements=structured_elements,  # 359 elements from previous step
    domain="research",  # Optional domain for context
    base_metadata=metadata
)
```

**LLM Enhancement**:
- Synthesizes rich titles for each memory
- Adds semantic metadata (keywords, topics, categories)
- Identifies relationships between elements
- Generates contextual summaries
- Maintains source provenance (chunk_index, block_index, provider)

**Fallback**: If LLM fails, falls back to `MemoryTransformer.content_element_to_memory_request` for deterministic conversion.

### 4. Batch Memory Creation (`create_hierarchical_memory_batch`)

**Input**: List of `AddMemoryRequest` objects (from either Simple or Complex path)
**Output**: Batch creation result

```python
batch_request = BatchMemoryRequest(
    memories=memory_requests,  # 359 rich memories
    organization_id=org_id,
    namespace_id=ns_id,
    batch_size=20
)

# Reuses existing Temporal batch workflow
batch_result = await process_batch_with_temporal(batch_request, auth_response)
```

### 5. Linking Memories to Post (`link_batch_memories_to_post`)

**Input**: `upload_id`, `post_id`
**Output**: Linked memory count

```python
# Query Parse for all Memory objects with this upload_id
memories = await parse_query("Memory", where={"upload_id": upload_id})

# Link them all to the Post
await parse_integration.link_memories_to_post(post_id, memory_object_ids)
```

## Provider Support

### Reducto
- SDK: `reducto` package with typed Pydantic models
- Structure: `PipelineResponse.result.parse.result.chunks[]`
- Types: Text, Figure, Table, Heading, List
- Confidence: String values ("high", "medium", "low")

### TensorLake
- Uses `ProviderContentExtractor.extract_from_tensorlake`
- Returns list of `ContentElement` objects

### Gemini, DeepSeek-OCR, PaddleOCR
- Custom Pydantic models in `models/provider_types/`
- Adapters in `core/document_processing/provider_adapter.py`

## Key Design Decisions

### 1. Post Reference Pattern
**Problem**: Provider JSON can be 10MB+ (Temporal workflow history has 4MB limit)
**Solution**: Store provider JSON in Parse Post, pass only `post_id` reference between activities

### 2. Typed SDK Parsing
**Problem**: Manual JSON navigation is error-prone and brittle
**Solution**: Use provider SDKs' Pydantic types for type-safe traversal (e.g., Reducto's `PipelineResponse`)

### 3. Simple vs Complex Branching
**Problem**: Simple documents don't need expensive LLM processing
**Solution**: Analyze structure upfront, route to appropriate path
- Simple: Direct markdown chunking (fast, cheap)
- Complex: LLM synthesis (slow, expensive, high-quality)

### 4. Reusable Components
- `fetch_post_with_provider_result_async`: Centralized Parse Post fetching with file decompression
- `parse_with_provider_sdk`: Unified typed parsing for all providers
- `provider_adapter.extract_structured_elements`: Provider-agnostic element extraction

## Testing

### Unit Tests
```bash
# Test with real 30-page Reducto file
poetry run pytest tests/test_extract_structured_content_real.py -v -s

# Expected output:
# ✓ 96 chunks processed
# ✓ 359 elements extracted (342 text + 10 images + 7 tables)
# ✓ All elements have proper metadata and types
```

### Integration Tests
```bash
# Full end-to-end document processing
poetry run pytest tests/test_document_processing_v2.py -v
```

## Monitoring & Logging

Each activity logs:
- Decision path taken (simple/complex)
- Element counts by type
- Processing time
- Provider-specific metadata

Example logs:
```
INFO: Structure analysis decision: complex | analysis={'has_tables': True, 'has_images': True}
INFO: Processing Reducto response with 96 chunks
INFO: Extracted 359 content elements from Reducto response
INFO: Complex path (LLM): generated 359 memory requests
```

## Future Enhancements

1. **Semantic Chunking**: Use embeddings to create semantically coherent chunks across element boundaries
2. **Table Structure Parsing**: Parse table HTML/markdown into structured_data with headers and rows
3. **Image OCR Integration**: Extract text from images for richer search
4. **Cross-document Relationships**: Link related concepts across multiple documents
5. **Incremental Updates**: Update only changed sections when documents are re-processed

