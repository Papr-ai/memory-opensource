# Temporal Payload Limits & Parse Storage Strategy

## Problem Statement

Temporal has strict limits on workflow history and activity payloads:
- **Workflow History**: ~50MB total or 50,000 events
- **Activity Result Size**: Recommended < 2MB per activity
- **Our Use Case**: Processing large documents (1000+ pages) can produce **multiple MB** of structured elements

### Example: 30-Page Document
```
Input: Google research paper (30 pages)
Output: 359 content elements
  - 342 text blocks
  - 10 images
  - 7 tables
  
Estimated Size: ~500KB-1MB (within limits, but risky)
```

### Example: 1000-Page Document
```
Input: Technical manual (1000 pages)
Output: ~12,000+ content elements

Estimated Size: ~15-20MB ❌ EXCEEDS TEMPORAL LIMITS
```

## Solution: Hybrid Storage Pattern

Store large payloads in **Parse Server** (as compressed files), pass only **references** through Temporal workflow.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Activity 1: process_document_with_provider_from_reference  │
│  ✓ Stores: Full provider JSON as compressed file in Parse   │
│  ✓ Returns: Minimal payload with post_id reference          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓ (post_id reference, ~100 bytes)
┌─────────────────────────────────────────────────────────────┐
│  Activity 2: extract_structured_content_from_provider       │
│  ✓ Fetches: Provider JSON from Parse using post_id          │
│  ✓ Processes: Extracts 359-12,000+ content elements         │
│  ✓ Checks: Payload size > 500KB threshold?                  │
│    ├─ NO:  Returns full payload (small doc)                 │
│    └─ YES: Stores extraction in Parse, returns reference    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓ (reference or small payload)
┌─────────────────────────────────────────────────────────────┐
│  Activity 3: generate_llm_optimized_memory_structures       │
│  ✓ If stored: Fetches extraction from Parse using post_id   │
│  ✓ If inline: Uses provided structured_elements directly    │
│  ✓ Processes: Generates LLM-optimized memories              │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Details

### 1. Storage Functions (services/memory_management.py)

#### Store Extraction Result
```python
async def store_extraction_result_in_post(
    post_id: str,
    structured_elements: List[Dict[str, Any]],
    memory_requests: List[Dict[str, Any]],
    element_summary: Dict[str, int],
    decision: str
) -> str:
    """
    Store large extraction results in Parse Post.
    
    1. Compresses extraction data with gzip
    2. Uploads as Parse File (extraction_<post_id>_<uuid>.json.gz)
    3. Updates Post with extractionResultFile pointer
    4. Returns extraction filename
    """
```

**Compression Ratio**: Typically 5-10x reduction
- 1MB JSON → 100-200KB compressed
- 10MB JSON → 1-2MB compressed

#### Fetch Extraction Result
```python
async def fetch_extraction_result_from_post(
    post_id: str
) -> Optional[Dict[str, Any]]:
    """
    Fetch and decompress extraction result from Parse Post.
    
    1. Fetches Post to get extractionResultFile
    2. Downloads compressed file from URL
    3. Decompresses with gzip
    4. Returns full extraction data
    """
```

### 2. Activity: extract_structured_content_from_provider

**Adaptive Payload Strategy**:

```python
# Calculate payload size
extraction_size_estimate = len(str(structured_elements)) + len(str(memory_requests))
should_store_in_parse = extraction_size_estimate > 500_000  # 500KB threshold

if should_store_in_parse and post_id:
    # Store in Parse, return minimal reference
    extraction_result_id = await store_extraction_result_in_post(
        post_id=post_id,
        structured_elements=[elem.model_dump() for elem in structured_elements],
        memory_requests=[req.model_dump() for req in memory_requests],
        element_summary=element_types,
        decision="complex"
    )
    
    return {
        "decision": "complex",
        "extraction_stored": True,        # ✓ Flag for next activity
        "extraction_result_id": extraction_result_id,
        "post_id": post_id,               # ✓ Reference for fetch
        "element_summary": element_types,
        "structure_analysis": analysis,
        "provider": provider_name,
        "extraction_size": extraction_size_estimate
    }
else:
    # Small document, return inline
    return {
        "decision": "complex",
        "structured_elements": [elem.model_dump() for elem in structured_elements],
        "memory_requests": [req.model_dump() for req in memory_requests],
        "extraction_stored": False
    }
```

### 3. Activity: generate_llm_optimized_memory_structures

**Conditional Fetch**:

```python
async def generate_llm_optimized_memory_structures(
    content_elements: List[Dict[str, Any]],  # Empty if extraction_stored=True
    domain: Optional[str],
    base_metadata: MemoryMetadata,
    organization_id: str,
    namespace_id: str,
    use_llm: bool = True,
    post_id: Optional[str] = None,           # ✓ For fetching stored extraction
    extraction_stored: bool = False           # ✓ Flag to fetch from Parse
):
    # If extraction was stored (large document), fetch it
    if extraction_stored and post_id:
        extraction_data = await fetch_extraction_result_from_post(post_id)
        content_elements = extraction_data.get("structured_elements", [])
        logger.info(f"Fetched {len(content_elements)} elements from stored extraction")
    
    # Proceed with LLM generation using fetched or inline elements
    # ...
```

### 4. Workflow: document_processing.py

**Pass Storage Flags**:

```python
# After extraction activity
decision = extraction.get("decision", "complex")
extraction_stored = extraction.get("extraction_stored", False)
extraction_post_id = extraction.get("post_id", post_id_ref)

workflow.logger.info(f"extraction_stored={extraction_stored}")

# Complex path with conditional fetch
llm_gen = await workflow.execute_activity(
    "generate_llm_optimized_memory_structures",
    args=[
        extraction.get("structured_elements", []),  # Empty if stored
        getattr(metadata, "domain", None),
        metadata,
        organization_id,
        namespace_id,
        True,                    # use_llm
        extraction_post_id,      # ✓ post_id for fetch
        extraction_stored        # ✓ flag to fetch
    ],
    start_to_close_timeout=timedelta(minutes=20)
)
```

## Parse Server Schema Updates

### Post Class Fields

```javascript
{
  // Existing fields...
  
  // Provider result storage (from step 1)
  providerResultFile: File,  // Compressed provider JSON (e.g., reducto_output.json.gz)
  
  // Extraction result storage (from step 2)
  extractionResultFile: File,  // Compressed extraction (e.g., extraction_<uuid>.json.gz)
  extractionMetadata: {
    decision: "simple" | "complex",
    element_summary: {
      text: 342,
      image: 10,
      table: 7
    },
    total_elements: 359,
    total_memory_requests: 359,
    compressed_size: 125000,     // bytes
    original_size: 950000,        // bytes
    extracted_at: "2025-10-20T16:00:00Z"
  }
}
```

### Pydantic Model Updates (models/parse_server.py)

```python
class PostParseServer(BaseModel):
    # ... existing fields ...
    
    # Provider result storage (for large JSON files)
    provider_result_file: Optional[ParseFile] = None
    providerResultFile: Optional[ParseFile] = None  # camelCase alias
    
    # Extraction result storage (for large extraction results)
    extraction_result_file: Optional[ParseFile] = None
    extractionResultFile: Optional[ParseFile] = None  # camelCase alias
    extractionMetadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
```

## Benefits

### 1. **Scalability**
- ✓ Handles documents of any size (1-10,000+ pages)
- ✓ No Temporal history bloat
- ✓ Workflow history stays small (~KB instead of ~MB)

### 2. **Performance**
- ✓ Compression reduces storage/bandwidth by 5-10x
- ✓ Async fetching doesn't block workflow
- ✓ Only fetches when needed (lazy loading)

### 3. **Cost Optimization**
- ✓ Small documents use inline payload (fast, no extra Parse calls)
- ✓ Large documents use Parse storage (avoids Temporal limits)
- ✓ Automatic threshold-based decision

### 4. **Reliability**
- ✓ Temporal workflow history stays under limits
- ✓ Parse Server handles large file storage
- ✓ Graceful fallback if storage fails (returns inline payload)

## Size Thresholds

| Document Size | Elements | Payload Size | Strategy | Temporal Payload |
|---------------|----------|--------------|----------|------------------|
| 1-10 pages    | 10-100   | ~50-200KB    | Inline   | ~50-200KB        |
| 10-50 pages   | 100-500  | ~200-800KB   | **Threshold** | ~200-800KB   |
| 50-100 pages  | 500-1000 | ~800KB-1.5MB | Parse Storage | ~1-2KB (ref) |
| 100-1000 pages| 1K-12K   | ~1.5-20MB    | Parse Storage | ~1-2KB (ref) |
| 1000+ pages   | 12K+     | ~20MB+       | Parse Storage | ~1-2KB (ref) |

**Current Threshold**: 500KB (conservative, can be adjusted)

## Monitoring & Observability

### Logs to Track

```python
# Activity logs
logger.info(f"Compressing extraction: {original_size} bytes → {compressed_size} bytes")
logger.info(f"Stored large extraction result ({size:,} bytes) in Post {post_id}")
logger.info(f"Fetched {element_count} elements from stored extraction")

# Workflow logs
workflow.logger.info(f"extraction_stored={True}, will fetch from Parse")
workflow.logger.info(f"Complex path: using inline elements (size under threshold)")
```

### Metrics to Monitor

1. **Payload Size Distribution**: Track how many documents exceed threshold
2. **Compression Ratio**: Monitor gzip effectiveness
3. **Fetch Latency**: Time to download/decompress from Parse
4. **Storage Cost**: Parse Server file storage usage

## Testing

### Unit Test: Payload Size Calculation

```python
@pytest.mark.asyncio
async def test_large_extraction_stored_in_parse():
    """Test that large extractions are stored in Parse"""
    
    # Create 1000+ fake elements (simulating large document)
    large_elements = [
        TextElement(
            element_id=f"elem_{i}",
            content="x" * 1000  # 1KB per element
        )
        for i in range(1000)
    ]
    
    result = await extract_structured_content_from_provider(
        provider_specific=large_provider_json,
        provider_name="reducto",
        base_metadata=metadata,
        organization_id="test_org",
        namespace_id="test_ns"
    )
    
    # Should be stored in Parse
    assert result["extraction_stored"] is True
    assert result["post_id"] is not None
    assert len(result["structured_elements"]) == 0  # Not inline
    assert result["extraction_size"] > 500_000
```

### Integration Test: Full Workflow

```python
@pytest.mark.asyncio
async def test_large_document_workflow():
    """Test end-to-end processing with Parse storage"""
    
    # 1. Process large document
    workflow_id = await start_document_workflow(large_file_ref)
    
    # 2. Wait for completion
    result = await wait_for_workflow(workflow_id)
    
    # 3. Verify memories were created
    assert result["total_memory_items"] > 0
    
    # 4. Verify Post has extraction stored
    post = await fetch_post(result["post_id"])
    assert post["extractionResultFile"] is not None
    assert post["extractionMetadata"]["total_elements"] > 1000
```

## Future Optimizations

1. **Streaming Extraction**: Process elements in batches instead of all at once
2. **Selective Fetch**: Only fetch needed elements for LLM (e.g., first 100)
3. **Caching**: Cache decompressed extraction in memory for repeated access
4. **Chunked Upload**: For very large extractions, use multipart upload
5. **TTL/Cleanup**: Auto-delete old extraction files after X days

## Migration Path

### Phase 1: Backwards Compatible (Current)
- ✓ Small documents use inline payload (no breaking changes)
- ✓ Large documents store in Parse (new feature)
- ✓ Both paths work seamlessly

### Phase 2: Always Store (Future)
- Store all extractions in Parse regardless of size
- Simplify code by removing inline path
- Reduce Temporal history size for all documents

### Phase 3: Streaming (Future)
- Process documents in streaming fashion
- Never hold full extraction in memory
- Ultimate scalability for massive documents

