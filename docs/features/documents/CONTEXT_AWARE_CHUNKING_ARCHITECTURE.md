# Context-Aware Document Chunking Architecture (2025)

**Version**: 2.0
**Date**: November 2025
**Status**: Implementation Plan
**Research Foundation**: Based on latest 2025 research (Vision-Guided Chunking ArXiv 2506.16035, Anthropic Contextual Retrieval, HiChunk, Late Chunking)

---

## Executive Summary

This document outlines a comprehensive architecture for context-aware document chunking that addresses critical issues in the current implementation:

**Current Problems**:
- ❌ Tables/images isolated from surrounding text (context loss)
- ❌ Multi-page tables split without header preservation
- ❌ LLM processes each chunk independently (no document awareness)
- ❌ "Semantic" chunking only uses length-based grouping (no real semantic similarity)

**Solution**: Three-tiered approach combining:
1. **Contextual Retrieval** (Anthropic) - Add surrounding context to chunks
2. **Vision-Guided Chunking** (ArXiv 2506.16035) - Use multimodal LLMs for boundary detection
3. **Late Chunking** - Embed full document then split with context preservation

**Expected Improvements**:
- 30-40% improvement in retrieval accuracy (based on research benchmarks)
- Better preservation of document structure and semantics
- Improved handling of tables, images, and multi-modal content

---

## Research Foundation

### 1. Vision-Guided Chunking (ArXiv 2506.16035, June 2025)

**Key Innovation**: Use Large Multimodal Models (LMMs) to process documents in batches while maintaining semantic coherence.

**Findings**:
- Process 4-page batches with cross-batch context preservation
- Achieves better accuracy than traditional vanilla RAG systems
- Successfully handles multi-page tables, embedded figures, procedural content
- Preserves hierarchical document organization

**Model Used in Research**: Gemini 2.5 Pro

### 2. Contextual Retrieval (Anthropic, 2024/2025)

**Key Innovation**: Add document-level context to each chunk before embedding.

**Example**:
```
Original chunk: "Revenue grew 25% YoY to $2.3B"

With context:
Document: Q4 2024 Financial Report
Section: Revenue Analysis
Page: 15

[Previous context: "The company saw strong performance across all segments..."]

Revenue grew 25% YoY to $2.3B

[Following context: "This growth was primarily driven by cloud services..."]
```

**Findings**:
- Preserves semantic coherence more effectively
- 400 characters of surrounding text is optimal
- Critical for tables and images

### 3. Late Chunking (2025)

**Key Innovation**: Process entire document through embedding model first, then split chunks while retaining global context.

**Approach**:
```python
# Traditional (bad):
embed(chunk1), embed(chunk2), embed(chunk3)  # Each chunk isolated

# Late Chunking (good):
full_embeddings = embed(full_document)  # All chunks see each other
chunk1_emb = full_embeddings[0:100]     # Extract with context
chunk2_emb = full_embeddings[100:200]   # Extract with context
```

**Findings**:
- Each chunk embedding contains document-level context
- Better retrieval performance
- Works well with voyage-context-3 and similar models

### 4. HiChunk Benchmark (September 2025)

**Key Innovation**: Hierarchical Chunking + Auto-Merge retrieval algorithm.

**Findings**:
- Small chunks for precise retrieval, auto-merge for complete context
- Maintains parent-child relationships between chunks
- Significantly enhances quality of chunking, retrieval, and responses

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DOCUMENT PROCESSING PIPELINE                  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
         ┌───────────────────────────────────────────┐
         │  STEP 1: PROVIDER EXTRACTION              │
         │  (Reducto, TensorLake, etc.)              │
         │  → Raw elements with types                │
         └───────────────────────────────────────────┘
                                 │
                                 ▼
         ┌───────────────────────────────────────────┐
         │  STEP 2: STRUCTURE ANALYSIS               │
         │  Decision: Simple vs Complex Path         │
         └───────────────────────────────────────────┘
                    │                        │
           Simple   │                        │  Complex
                    ▼                        ▼
    ┌─────────────────────┐    ┌─────────────────────────────────┐
    │  Markdown Chunking  │    │  CONTEXT-AWARE CHUNKING (NEW)   │
    │  (Size-based)       │    │  ┌──────────────────────────┐   │
    └─────────────────────┘    │  │ 2a. Vision-Guided Split  │   │
                               │  │  (Gemini 2.5 Flash)      │   │
                               │  │  - 4-page batches        │   │
                               │  │  - Preserve multi-page   │   │
                               │  │    tables/sections       │   │
                               │  └──────────────────────────┘   │
                               │              │                   │
                               │              ▼                   │
                               │  ┌──────────────────────────┐   │
                               │  │ 2b. Context Extraction   │   │
                               │  │  - 400 chars before      │   │
                               │  │  - 400 chars after       │   │
                               │  │  - Document metadata     │   │
                               │  └──────────────────────────┘   │
                               │              │                   │
                               │              ▼                   │
                               │  ┌──────────────────────────┐   │
                               │  │ 2c. Hierarchical Chunk   │   │
                               │  │  - Group by section      │   │
                               │  │  - Preserve hierarchy    │   │
                               │  │  - 1000-6000 chars       │   │
                               │  └──────────────────────────┘   │
                               └─────────────────────────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────────┐
                               │  STEP 3: LLM ENHANCEMENT (NEW)  │
                               │  ┌──────────────────────────┐   │
                               │  │ 3a. Document Context     │   │
                               │  │  Injection               │   │
                               │  │  - Title, summary        │   │
                               │  │  - Section context       │   │
                               │  │  - Surrounding chunks    │   │
                               │  └──────────────────────────┘   │
                               │              │                   │
                               │              ▼                   │
                               │  ┌──────────────────────────┐   │
                               │  │ 3b. Metadata Generation  │   │
                               │  │  (GPT-4o-mini/Gemini)    │   │
                               │  │  - Rich titles           │   │
                               │  │  - Topics, keywords      │   │
                               │  │  - Relationships         │   │
                               │  └──────────────────────────┘   │
                               └─────────────────────────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────────┐
                               │  STEP 4: LATE CHUNKING          │
                               │  EMBEDDING (NEW - FUTURE)       │
                               │  - Embed full document context  │
                               │  - Extract chunk embeddings     │
                               │  - Preserve global information  │
                               └─────────────────────────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────────┐
                               │  STEP 5: MEMORY CREATION        │
                               │  - Store in Qdrant/MongoDB      │
                               │  - Rich metadata for search     │
                               │  - Context-aware embeddings     │
                               └─────────────────────────────────┘
```

---

## Component Specifications

### 1. Vision-Guided Chunking Module

**Purpose**: Use multimodal LLM to identify semantic chunk boundaries while preserving document structure.

**Model Selection**: **Gemini 2.5 Flash** (Recommended)
- **Why Gemini 2.5 Flash**:
  - ✅ Handles multimodal inputs (text + layout)
  - ✅ 1M token context window (can process many pages)
  - ✅ Fast and cost-effective ($0.075/1M input tokens)
  - ✅ Proven in research (original paper used Gemini 2.5 Pro)
  - ✅ Already integrated in your system
  - ✅ Better than GPT-4o for vision tasks at lower cost

**Alternative**: Gemini 2.5 Pro for highest quality (2x cost but better accuracy)

**Input**:
```python
{
    "pages": [page1, page2, page3, page4],  # 4-page batch
    "previous_context": {
        "last_section": "Revenue Analysis",
        "last_chunk_summary": "Discussion of Q3 performance..."
    },
    "document_metadata": {
        "title": "Q4 2024 Financial Report",
        "total_pages": 21,
        "domain": "financial"
    }
}
```

**Output**:
```python
{
    "semantic_chunks": [
        {
            "chunk_id": "chunk_001",
            "pages": [1, 2],
            "content_type": "multi_page_section",
            "boundary_type": "section_break",
            "elements": [element_ids],
            "reasoning": "Complete section about revenue analysis",
            "preserve_table": {
                "table_id": "table_5",
                "spans_pages": [1, 2],
                "headers": [...],
                "strategy": "keep_complete"
            }
        }
    ],
    "context_for_next_batch": {
        "last_section": "Operating Expenses",
        "summary": "Detailed breakdown of operational costs..."
    }
}
```

**Implementation**:
```python
async def vision_guided_chunk(
    pages: List[Dict[str, Any]],
    previous_context: Optional[Dict[str, Any]] = None,
    document_metadata: Dict[str, Any] = {}
) -> Dict[str, Any]:
    """
    Use Gemini 2.5 Flash to identify semantic chunk boundaries.

    Process documents in 4-page batches with cross-batch context.
    """

    prompt = f"""
You are analyzing a document for semantic chunking. Your goal is to identify natural chunk boundaries while preserving document structure.

DOCUMENT CONTEXT:
- Title: {document_metadata.get('title', 'Unknown')}
- Domain: {document_metadata.get('domain', 'general')}
- Total Pages: {document_metadata.get('total_pages', 'Unknown')}

PREVIOUS CONTEXT (from last batch):
{json.dumps(previous_context, indent=2) if previous_context else 'Start of document'}

CURRENT BATCH: Pages {pages[0]['page_number']} to {pages[-1]['page_number']}

INSTRUCTIONS:
1. Identify semantic chunk boundaries (where topics/sections change)
2. PRESERVE multi-page tables (keep all rows with headers)
3. PRESERVE figures with their captions and surrounding explanatory text
4. GROUP related paragraphs that discuss the same concept
5. MAINTAIN section hierarchy (headers, subheaders)
6. For each chunk, specify:
   - Which pages/elements it includes
   - Why this is a good semantic boundary
   - Any special handling (e.g., multi-page table, figure with context)

Return JSON with semantic chunks and context for next batch.
"""

    # Use Gemini 2.5 Flash for vision-guided analysis
    response = await gemini_flash.generate_content([
        {"text": prompt},
        *[{"inline_data": {"mime_type": "application/pdf", "data": page['image']}}
          for page in pages]
    ])

    return json.loads(response.text)
```

**Performance Targets**:
- Process 4 pages in ~3-5 seconds
- Batch overlap: 1 page between batches for continuity
- Cost: ~$0.30 per 100-page document

---

### 2. Context Window Extraction Module

**Purpose**: Extract surrounding text context for tables, images, and other non-text elements.

**Configuration**:
```python
CONTEXT_CONFIG = {
    "context_chars_before": 400,  # Research-backed optimal size
    "context_chars_after": 400,
    "include_section_header": True,
    "include_document_metadata": True,
    "max_context_elements": 3  # Max elements to look back/forward
}
```

**Implementation**:
```python
def extract_element_with_context(
    elements: List[ContentElement],
    target_index: int,
    config: dict
) -> Dict[str, Any]:
    """
    Extract element with surrounding context.

    For tables/images, includes:
    - Text before (caption, introduction, explanation)
    - Text after (analysis, conclusions, references)
    - Document metadata (title, section, page)
    """

    element = elements[target_index]

    # Extract text before
    context_before = ""
    context_elements_before = []
    chars_collected = 0

    for i in range(target_index - 1, -1, -1):
        if elements[i].content_type == ContentType.TEXT:
            text = elements[i].content
            needed_chars = config['context_chars_before'] - chars_collected

            if len(text) <= needed_chars:
                context_before = text + "\n\n" + context_before
                context_elements_before.append(elements[i].element_id)
                chars_collected += len(text)
            else:
                # Take last N characters
                context_before = "..." + text[-needed_chars:] + "\n\n" + context_before
                context_elements_before.append(elements[i].element_id)
                break

            if chars_collected >= config['context_chars_before']:
                break
            if len(context_elements_before) >= config['max_context_elements']:
                break

    # Extract text after (similar logic)
    context_after = ""
    context_elements_after = []
    chars_collected = 0

    for i in range(target_index + 1, len(elements)):
        if elements[i].content_type == ContentType.TEXT:
            text = elements[i].content
            needed_chars = config['context_chars_after'] - chars_collected

            if len(text) <= needed_chars:
                context_after = context_after + "\n\n" + text
                context_elements_after.append(elements[i].element_id)
                chars_collected += len(text)
            else:
                # Take first N characters
                context_after = context_after + "\n\n" + text[:needed_chars] + "..."
                context_elements_after.append(elements[i].element_id)
                break

            if chars_collected >= config['context_chars_after']:
                break
            if len(context_elements_after) >= config['max_context_elements']:
                break

    # Build section context
    section_context = ""
    if config['include_section_header']:
        section_title = element.metadata.get('section_title', '')
        section_level = element.metadata.get('section_level', 1)
        if section_title:
            section_context = f"{'#' * section_level} {section_title}\n\n"

    return {
        "element": element,
        "context_before": context_before.strip(),
        "context_after": context_after.strip(),
        "section_context": section_context.strip(),
        "context_elements_before": context_elements_before,
        "context_elements_after": context_elements_after
    }
```

**Usage Example**:
```python
# For a table element
table_with_context = extract_element_with_context(elements, table_index, CONTEXT_CONFIG)

# Creates enriched content:
"""
Document: Q4 2024 Financial Report
Section: Revenue Analysis
Page: 15

[Context before: "The company achieved strong performance across all segments.
Table 1 below shows the revenue breakdown by product line for Q4 2024..."]

| Product Line | Q4 Revenue | YoY Growth |
|--------------|------------|------------|
| Cloud        | $1.2B      | 35%        |
| Enterprise   | $800M      | 18%        |
| Consumer     | $300M      | 12%        |

[Context after: "As shown in the table, Cloud services drove the majority of growth,
representing 52% of total revenue. This trend is expected to continue..."]
"""
```

---

### 3. Document Context Injection Module

**Purpose**: Inject document-level context into LLM prompts for metadata generation.

**Enhanced Prompt Template**:
```python
CONTEXT_AWARE_PROMPT = """
# DOCUMENT CONTEXT
Document Title: {document_title}
Document Type: {document_type}
Document Summary: {document_summary}
Total Pages: {total_pages}
Domain: {domain}

# SECTION CONTEXT
Current Section: {section_title} (Level {section_level})
Section Summary: {section_summary}
Page Number: {page_number}

# SURROUNDING CHUNKS
Previous Chunk Summary: {previous_chunk_summary}
Next Chunk Summary: {next_chunk_summary}

# CONTENT TO ANALYZE
Content Type: {content_type}
{context_before}

{content}

{context_after}

---

TASK: Analyze the content above and generate rich metadata for searchability.

IMPORTANT:
1. Consider this content in the context of the overall document "{document_title}"
2. Identify how this content relates to the document's main themes
3. Create natural language queries users might ask about this content
4. Preserve relationships to surrounding content
5. DO NOT modify the actual content - only generate metadata

Generate metadata as JSON:
{{
    "title": "Descriptive title considering document context",
    "topics": ["topic1", "topic2", "topic3"],
    "keywords": ["key1", "key2"],
    "entities": ["entity1", "entity2"],
    "semantic_relationships": [
        "related_to_section_X",
        "supports_claim_in_chunk_Y"
    ],
    "query_patterns": [
        "Natural language question 1",
        "Natural language question 2"
    ],
    "document_position": {{
        "section": "{section_title}",
        "semantic_role": "introduction|analysis|conclusion|supporting_data"
    }}
}}
"""
```

**Implementation**:
```python
async def generate_contextual_metadata(
    element_with_context: Dict[str, Any],
    document_metadata: Dict[str, Any],
    section_metadata: Dict[str, Any],
    surrounding_chunks: Dict[str, str]
) -> Dict[str, Any]:
    """
    Generate metadata with full document context awareness.
    """

    prompt = CONTEXT_AWARE_PROMPT.format(
        # Document context
        document_title=document_metadata.get('title', 'Unknown'),
        document_type=document_metadata.get('type', 'document'),
        document_summary=document_metadata.get('summary', ''),
        total_pages=document_metadata.get('total_pages', 'Unknown'),
        domain=document_metadata.get('domain', 'general'),

        # Section context
        section_title=section_metadata.get('title', 'Unknown'),
        section_level=section_metadata.get('level', 1),
        section_summary=section_metadata.get('summary', ''),
        page_number=element_with_context['element'].metadata.get('page_number', 'Unknown'),

        # Surrounding context
        previous_chunk_summary=surrounding_chunks.get('previous', ''),
        next_chunk_summary=surrounding_chunks.get('next', ''),

        # Content
        content_type=element_with_context['element'].content_type.value,
        context_before=element_with_context['context_before'],
        content=element_with_context['element'].content,
        context_after=element_with_context['context_after']
    )

    # Call LLM (GPT-4o-mini or Gemini Flash)
    response = await llm.generate(prompt)

    return json.loads(response)
```

---

### 4. Table Header Preservation Module

**Purpose**: Split large tables while preserving headers and structure.

**Strategy**:
```python
def split_large_table_with_context(
    table_element: TableElement,
    context_before: str,
    context_after: str,
    max_chunk_size: int = 6000
) -> List[ContentElement]:
    """
    Split large tables while preserving:
    1. Headers in each chunk
    2. Surrounding context (table caption, analysis)
    3. Table structure and formatting
    """

    if not hasattr(table_element, 'structured_data') or not table_element.structured_data:
        # Fall back to text-based splitting
        return split_element_semantically(table_element)

    headers = table_element.structured_data.get('headers', [])
    rows = table_element.structured_data.get('rows', [])

    # Calculate header size
    header_text = " | ".join(str(h) for h in headers)
    header_size = len(header_text) + 100  # Include formatting overhead

    # Calculate context size
    context_size = len(context_before) + len(context_after) + 200

    # Available size for rows
    available_size = max_chunk_size - header_size - context_size

    if available_size < 500:
        logger.warning(f"Very large headers/context, may produce small chunks")
        available_size = 500

    # Group rows into chunks
    chunks = []
    current_rows = []
    current_size = 0

    for row in rows:
        row_text = " | ".join(str(cell) for cell in row)
        row_size = len(row_text) + 10  # Include newline overhead

        if current_size + row_size > available_size and current_rows:
            # Create chunk with headers and context
            chunk = create_table_chunk_with_context(
                headers=headers,
                rows=current_rows,
                context_before=context_before,
                context_after=context_after,
                chunk_index=len(chunks),
                total_chunks="TBD",  # Will update after
                original_table_id=table_element.element_id
            )
            chunks.append(chunk)
            current_rows = []
            current_size = 0

        current_rows.append(row)
        current_size += row_size

    # Final chunk
    if current_rows:
        chunk = create_table_chunk_with_context(
            headers=headers,
            rows=current_rows,
            context_before=context_before,
            context_after=context_after,
            chunk_index=len(chunks),
            total_chunks="TBD",
            original_table_id=table_element.element_id
        )
        chunks.append(chunk)

    # Update total_chunks metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_index'] = i
        chunk.metadata['total_chunks'] = len(chunks)
        chunk.metadata['is_table_chunk'] = True
        chunk.metadata['original_table_id'] = table_element.element_id

    logger.info(f"Split large table into {len(chunks)} chunks with preserved headers")
    return chunks


def create_table_chunk_with_context(
    headers: List[str],
    rows: List[List[str]],
    context_before: str,
    context_after: str,
    chunk_index: int,
    total_chunks: str,
    original_table_id: str
) -> TableElement:
    """
    Create a table chunk with headers and context.
    """

    # Build table text
    table_text = ""

    # Add context before
    if context_before:
        table_text += context_before + "\n\n"

    # Add table
    table_text += "| " + " | ".join(str(h) for h in headers) + " |\n"
    table_text += "|" + "|".join(["---"] * len(headers)) + "|\n"

    for row in rows:
        table_text += "| " + " | ".join(str(cell) for cell in row) + " |\n"

    # Add chunk indicator
    table_text += f"\n[Table chunk {chunk_index + 1} of {total_chunks}]\n"

    # Add context after
    if context_after and chunk_index == 0:  # Only add after context to first chunk
        table_text += "\n" + context_after

    # Create TableElement
    chunk = TableElement(
        element_id=f"{original_table_id}_chunk_{chunk_index}",
        content=table_text.strip(),
        structured_data={
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "column_count": len(headers),
            "is_chunk": True,
            "chunk_index": chunk_index,
            "original_table_id": original_table_id
        },
        headers=headers,
        rows=rows,
        metadata={
            "is_table_chunk": True,
            "chunk_index": chunk_index,
            "has_context_before": bool(context_before),
            "has_context_after": bool(context_after and chunk_index == 0)
        }
    )

    return chunk
```

---

### 5. Late Chunking Embedding Module (Phase 3 - Future)

**Purpose**: Generate embeddings with full document context, then extract chunk embeddings.

**Model Recommendation**: Continue with Qwen 3 4B (2650-dim embeddings)

**Implementation** (Future):
```python
async def late_chunking_embed(
    document_content: str,
    chunk_boundaries: List[Tuple[int, int]],  # [(start, end), ...]
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
) -> List[List[float]]:
    """
    Generate embeddings with late chunking approach.

    1. Embed entire document to get token-level embeddings
    2. Extract chunk embeddings with document context
    """

    # Tokenize full document
    tokens = tokenizer.encode(document_content)

    # Get token-level embeddings (each token "sees" entire document during encoding)
    full_doc_embeddings = await qwen_embed_tokens(tokens)

    # Extract chunk embeddings
    chunk_embeddings = []
    for start, end in chunk_boundaries:
        # Average token embeddings within chunk
        chunk_emb = np.mean(full_doc_embeddings[start:end], axis=0)
        chunk_embeddings.append(chunk_emb.tolist())

    return chunk_embeddings
```

---

## Implementation Phases

### Phase 1: Context Windows & Document Injection (Week 1)

**Goals**:
- Add context extraction for tables/images (400 chars before/after)
- Inject document context into LLM prompts
- Update prompt templates with context awareness

**Files to Modify**:
1. `/core/document_processing/hierarchical_chunker.py`
   - Add `extract_element_with_context()` function
   - Update `_semantic_chunking()` to use context

2. `/core/document_processing/llm_memory_generator.py`
   - Add document metadata parameters
   - Update all prompt templates
   - Inject context into LLM calls

3. `/cloud_plugins/temporal/activities/document_activities.py`
   - Pass document metadata to LLM generator
   - Collect and pass surrounding chunk summaries

**Success Metrics**:
- Tables/images have 400 chars of surrounding context
- LLM prompts include document title, section, and page
- Metadata reflects document-level themes

### Phase 2: Vision-Guided Chunking (Week 2)

**Goals**:
- Implement Gemini 2.5 Flash integration
- Process documents in 4-page batches
- Preserve multi-page tables and sections

**Files to Create**:
1. `/core/document_processing/vision_guided_chunker.py`
   - Main vision-guided chunking logic
   - Gemini API integration
   - Batch processing with overlap

**Files to Modify**:
1. `/cloud_plugins/temporal/workflows/document_processing.py`
   - Add vision-guided chunking option
   - Route complex documents to vision chunker

**Success Metrics**:
- Multi-page tables preserved intact
- Section boundaries correctly identified
- Cross-page context maintained

### Phase 3: Table Header Preservation (Week 2-3)

**Goals**:
- Split large tables while preserving headers
- Maintain table structure across chunks
- Include context in each chunk

**Files to Modify**:
1. `/core/document_processing/hierarchical_chunker.py`
   - Add `split_large_table_with_context()`
   - Update table handling in chunking strategies

**Success Metrics**:
- Large tables split with headers in each chunk
- Table chunks maintain context
- Structure preserved for retrieval

### Phase 4: Late Chunking (Week 4 - Future)

**Goals**:
- Implement late chunking with Qwen embeddings
- Generate document-aware embeddings
- Improve retrieval performance

**Files to Create**:
1. `/core/document_processing/late_chunking_embedder.py`
   - Token-level embedding generation
   - Chunk extraction with context

**Success Metrics**:
- Document-aware embeddings generated
- Improved retrieval accuracy
- Better semantic search results

---

## Testing Strategy

### Unit Tests

```python
# Test context extraction
def test_context_extraction():
    elements = create_test_elements()
    table_index = 5

    result = extract_element_with_context(elements, table_index, CONTEXT_CONFIG)

    assert len(result['context_before']) <= 400
    assert len(result['context_after']) <= 400
    assert result['element'].element_id == elements[table_index].element_id

# Test table splitting with headers
def test_table_split_with_headers():
    large_table = create_large_table_element(rows=1000)
    context_before = "This table shows revenue breakdown..."
    context_after = "As shown above, cloud revenue dominates..."

    chunks = split_large_table_with_context(
        large_table, context_before, context_after, max_chunk_size=6000
    )

    assert len(chunks) > 1
    for chunk in chunks:
        assert 'headers' in chunk.structured_data
        assert len(chunk.structured_data['headers']) == len(large_table.headers)
        assert 'revenue breakdown' in chunk.content.lower()

# Test vision-guided chunking
async def test_vision_guided_chunking():
    pages = load_test_pdf_pages(num_pages=4)

    result = await vision_guided_chunk(pages, None, {"title": "Test Doc"})

    assert 'semantic_chunks' in result
    assert len(result['semantic_chunks']) > 0
    assert 'context_for_next_batch' in result
```

### Integration Tests

```python
async def test_full_context_aware_pipeline():
    """Test complete pipeline with context-aware chunking"""

    # Upload test document (21-page PDF with tables and images)
    response = await upload_document_v2(test_pdf, hierarchical_enabled=True)

    upload_id = response['upload_id']
    workflow_id = response['workflow_id']

    # Wait for workflow completion
    await wait_for_workflow(workflow_id, timeout=300)

    # Fetch created memories
    memories = await fetch_memories_by_upload_id(upload_id)

    # Assertions
    assert len(memories) < 96  # Should be fewer chunks with better grouping
    assert len(memories) >= 15  # Should have reasonable number of chunks

    # Check context preservation
    table_memories = [m for m in memories if 'table' in m.content.lower()]
    for table_mem in table_memories:
        # Should have context before/after
        assert len(table_mem.content) > len(table_mem.metadata.get('original_table_content', ''))

    # Check document context injection
    for memory in memories:
        assert 'document_title' in memory.metadata
        assert 'section_title' in memory.metadata
        assert 'page_number' in memory.metadata
```

### Performance Benchmarks

```python
async def benchmark_chunking_strategies():
    """Compare chunking strategies on retrieval accuracy"""

    test_documents = load_test_corpus()
    test_queries = load_test_queries()

    strategies = [
        "legacy_semantic",
        "hierarchical_without_context",
        "context_aware",
        "vision_guided"
    ]

    results = {}

    for strategy in strategies:
        # Process documents
        chunks = await process_documents(test_documents, strategy=strategy)

        # Embed and store
        await embed_and_store(chunks)

        # Run retrieval tests
        precision = []
        recall = []

        for query, expected_docs in test_queries:
            retrieved = await search_memories(query, top_k=5)
            p, r = calculate_precision_recall(retrieved, expected_docs)
            precision.append(p)
            recall.append(r)

        results[strategy] = {
            "avg_precision": np.mean(precision),
            "avg_recall": np.mean(recall),
            "f1_score": 2 * np.mean(precision) * np.mean(recall) / (np.mean(precision) + np.mean(recall))
        }

    # Print comparison
    print(f"\nChunking Strategy Comparison:\n{json.dumps(results, indent=2)}")

    # Expected results based on research:
    # Context-aware should show 30-40% improvement over legacy
```

---

## Configuration

```python
# /config/chunking_config.py

CHUNKING_CONFIG = {
    # Context window settings
    "context_extraction": {
        "enabled": True,
        "chars_before": 400,
        "chars_after": 400,
        "max_context_elements": 3,
        "include_section_header": True
    },

    # Vision-guided chunking settings
    "vision_guided": {
        "enabled": True,  # Enable for complex documents
        "model": "gemini-2.5-flash",  # or "gemini-2.5-pro" for higher quality
        "batch_size_pages": 4,
        "batch_overlap_pages": 1,
        "min_confidence": 0.7,
        "fallback_to_hierarchical": True  # If vision fails
    },

    # Hierarchical chunking settings
    "hierarchical": {
        "enabled": True,
        "max_chunk_size": 6000,
        "min_chunk_size": 1000,
        "overlap_size": 200,
        "preserve_tables": True,
        "preserve_images": True,
        "group_by_section": True
    },

    # Document context injection
    "document_context": {
        "enabled": True,
        "include_document_summary": True,
        "include_section_context": True,
        "include_surrounding_chunks": True,
        "max_summary_length": 500
    },

    # Table handling
    "table_processing": {
        "preserve_headers": True,
        "split_large_tables": True,
        "max_table_chunk_size": 6000,
        "include_table_context": True
    },

    # LLM enhancement
    "llm_enhancement": {
        "enabled": True,
        "primary_model": "gpt-4o-mini",
        "fallback_models": ["gemini-2.5-flash", "groq-llama"],
        "batch_size": 10,
        "include_document_context": True
    },

    # Late chunking (future)
    "late_chunking": {
        "enabled": False,  # Not yet implemented
        "embedding_model": "Qwen/Qwen3-Embedding-4B",
        "use_full_document_context": True
    }
}
```

---

## Cost Analysis

### Gemini 2.5 Flash (Vision-Guided Chunking)

**Pricing**:
- Input: $0.075 / 1M tokens
- Output: $0.30 / 1M tokens

**Estimate for 100-page document**:
- Process in 25 batches (4 pages each)
- ~2K tokens per page = 8K input tokens per batch
- ~500 output tokens per batch (chunk boundaries JSON)
- Total: 200K input + 12.5K output tokens
- Cost: (200K × $0.075 + 12.5K × $0.30) / 1M = **$0.019 per 100-page document**

### GPT-4o-mini (Metadata Generation)

**Pricing**:
- Input: $0.150 / 1M tokens
- Output: $0.600 / 1M tokens

**Estimate for 100-page document** (20-25 chunks):
- ~3K tokens per chunk for prompt + context
- ~300 output tokens per chunk (metadata JSON)
- Total: 75K input + 7.5K output tokens
- Cost: (75K × $0.150 + 7.5K × $0.600) / 1M = **$0.016 per 100-page document**

**Total Cost per 100-page document**: ~$0.035 (3.5 cents)

**Comparison**:
- Legacy approach (no context): $0.010 per 100-page document
- New approach (context-aware): $0.035 per 100-page document
- **Additional cost**: $0.025 per 100-page document (250% increase)
- **Expected improvement**: 30-40% better retrieval accuracy (research-backed)

**ROI**: For most use cases, the improved retrieval accuracy far outweighs the marginal cost increase.

---

## Monitoring & Metrics

### Key Performance Indicators (KPIs)

1. **Chunking Quality**:
   - Average chunk size (target: 2000-4000 chars)
   - Chunks per document (target: 15-25 for 21-page doc)
   - Tables with preserved headers (target: 100%)
   - Elements with context (target: 100% for tables/images)

2. **Retrieval Accuracy**:
   - Precision@5 (target: >80%)
   - Recall@10 (target: >90%)
   - F1-score (target: >0.85)
   - Context relevance score (target: >0.8)

3. **Performance**:
   - Vision-guided chunking time (target: <5s per 4 pages)
   - LLM metadata generation time (target: <2s per chunk)
   - End-to-end processing time (target: <5 min for 100-page doc)

4. **Cost**:
   - Cost per document (target: <$0.05 per 100 pages)
   - LLM API costs (track daily/monthly spend)

### Logging

```python
logger.info(f"Context extraction complete: {len(elements_with_context)} elements")
logger.info(f"Tables with context: {tables_with_context}/{total_tables}")
logger.info(f"Vision-guided chunking: {len(semantic_chunks)} chunks identified")
logger.info(f"Document context injection: {successful_injections}/{total_chunks}")
logger.info(f"Chunk size distribution: min={min_size}, max={max_size}, avg={avg_size}")
```

### Alerts

- Alert if chunk size > 8000 chars (may exceed embedding limits)
- Alert if >50% of chunks missing context
- Alert if vision-guided chunking fails >10% of batches
- Alert if LLM metadata generation fails >20% of chunks

---

## Migration Plan

### From Current System to Context-Aware

**Step 1**: Deploy Phase 1 (Context Windows) - **Low Risk**
- Backward compatible
- No breaking changes
- Can run in parallel with existing system

**Step 2**: Enable for New Documents - **Low Risk**
- Set feature flag `context_aware_chunking=True` for new uploads
- Monitor performance and cost
- Compare retrieval quality

**Step 3**: Backfill Existing Documents - **Medium Risk**
- Re-process high-value documents with new system
- Compare old vs new chunk quality
- Migrate incrementally

**Step 4**: Deploy Phase 2 (Vision-Guided) - **Medium Risk**
- Enable only for complex documents (tables, images)
- Monitor Gemini API costs and latency
- Validate multi-page table preservation

**Step 5**: Full Production Rollout - **Low Risk**
- Enable for all documents
- Deprecate legacy chunking
- Monitor KPIs and cost

---

## Future Enhancements

1. **Adaptive Chunking** (Q1 2026):
   - LLM decides optimal chunk size per document
   - Dynamic sizing based on content density

2. **Cross-Document Linking** (Q2 2026):
   - Identify relationships between documents
   - Build knowledge graph from document corpus

3. **Streaming Chunking** (Q3 2026):
   - Process documents incrementally
   - Real-time chunk updates

4. **Multi-Language Support** (Q4 2026):
   - Extend to non-English documents
   - Language-aware context extraction

---

## References

1. **Vision-Guided Chunking**: ArXiv 2506.16035 (June 2025)
2. **Contextual Retrieval**: Anthropic Blog (2024/2025)
3. **HiChunk Benchmark**: ArXiv 2509.11552 (September 2025)
4. **Late Chunking**: KX Systems Blog (2025)
5. **voyage-context-3**: Voyage AI Blog (July 2025)
6. **Max-Min Semantic Chunking**: Springer (2025)

---

## Appendix: Code Locations

| Component | File Path | Lines |
|-----------|-----------|-------|
| Current Semantic Chunking | `/core/document_processing/hierarchical_chunker.py` | 394-467 |
| Current Hierarchical Chunking | `/core/document_processing/hierarchical_chunker.py` | 504-593 |
| LLM Memory Generator | `/core/document_processing/llm_memory_generator.py` | 763-829 |
| Chunk Activity | `/cloud_plugins/temporal/activities/document_activities.py` | 1652-1850 |
| Document Workflow | `/cloud_plugins/temporal/workflows/document_processing.py` | 145-180 |
| Provider Adapter | `/core/document_processing/provider_adapter.py` | 14-100 |

---

**Document Version**: 2.0
**Last Updated**: November 2025
**Status**: Ready for Implementation
**Estimated Implementation Time**: 3-4 weeks (Phases 1-3)
