# Sync Tiers Embedding Improvements

## Summary

Fixed server-side embedding enrichment to provide embeddings for **both Tier0 AND Tier1** items, significantly reducing client-side embedding generation overhead.

---

## Changes Made

### 1. **Enable Tier0 Embedding Enrichment** (`routers/v1/sync_routes.py`)

**Before**: Tier0 embeddings were explicitly skipped with a comment saying "embedding support for tier0 would require extending the Memory model"

**After**: Tier0 now uses the same `enrich_memories_with_embeddings_batch()` function as Tier1

```python
# Lines 302-324
if sync_request.include_embeddings and tier0_items:
    logger.info(f"Enriching {len(tier0_items)} Tier0 items with Qdrant embeddings...")
    try:
        items_to_enrich = tier0_items[:sync_request.embed_limit] if sync_request.embed_limit > 0 else tier0_items
        should_quantize = sync_request.embedding_format != EmbeddingFormat.FLOAT32
        enriched_items = await enrich_memories_with_embeddings_batch(
            items_to_enrich,
            memory_graph,
            quantize=should_quantize
        )
        tier0_items[:len(enriched_items)] = enriched_items
        # ... logging ...
```

### 2. **Improve Tier1 Embedding Enrichment** (`routers/v1/sync_routes.py`)

**Before**: No `embed_limit` enforcement, no detailed logging of enrichment success rate

**After**: 
- Respects `embed_limit` parameter
- Logs enrichment statistics
- Better error handling

```python
# Lines 262-285
items_to_enrich = tier1_items[:sync_request.embed_limit] if sync_request.embed_limit > 0 else tier1_items
enriched_items = await enrich_memories_with_embeddings_batch(...)
tier1_items[:len(enriched_items)] = enriched_items

# Count how many actually have embeddings
enriched_count = sum(1 for item in enriched_items if getattr(item, embedding_field, None) is not None)
logger.info(f"Tier1 embedding enrichment complete: {enriched_count}/{len(items_to_enrich)} items enriched")
```

### 3. **New Comprehensive Test** (`tests/test_sync_v1.py`)

Added `test_v1_sync_tiers_qwen4b_coreml_full_precision()` which validates:

- ✅ Both Tier0 AND Tier1 get embeddings
- ✅ Embeddings are 2560 dimensions (Qwen4B)
- ✅ Embeddings are float32 format (for CoreML/ANE)
- ✅ At least 50% coverage from Qdrant cache
- ✅ No INT8 quantization when float32 requested
- ✅ Proper type checking (float, not int)

---

## Expected Behavior

### Before Fix

```
Tier0: 200 items
  - Server: 0 embeddings ❌
  - SDK local generation: 200 embeddings (29s)

Tier1: 200 items
  - Server: 120 embeddings ✅
  - SDK local generation: 80 embeddings (12s)

Total: 280 local embeddings, ~41s initialization
```

### After Fix

```
Tier0: 200 items
  - Server: 150+ embeddings ✅ (from Qdrant cache)
  - SDK local generation: 50 embeddings (7s)

Tier1: 200 items
  - Server: 150+ embeddings ✅ (from Qdrant cache)
  - SDK local generation: 50 embeddings (7s)

Total: 100 local embeddings, ~14s initialization
Savings: 180 embeddings, ~27 seconds 🚀
```

**Note**: Actual numbers depend on how many memories exist in Qdrant. Newly created memories won't have cached embeddings until they're indexed.

---

## Testing

Run the new test:

```bash
cd /Users/shawkatkabbara/Documents/GitHub/memory
pytest tests/test_sync_v1.py::test_v1_sync_tiers_qwen4b_coreml_full_precision -v
```

Expected output:

```
================================================================================
QWEN4B COREML FULL PRECISION TEST (2560-dim float32)
================================================================================

--- TIER0 EMBEDDINGS ---
Total Tier0 items returned: 100
  Item 1 (id=584d7542-2b11-4bae-...)
    - Dimension: 2560 ✓
    - Type: float ✓
    - Sample values: [-0.00023892, 0.05796862, 0.00587674, ...]
    - Has INT8: False ✓
  ...

Tier0 Summary:
  - Items with embeddings: 85/100
  - Correct dimension (2560): 85/85
  - Correct type (float32): 85/85

--- TIER1 EMBEDDINGS ---
Total Tier1 items returned: 100
  Item 1 (id=413ce029-2e08-488c-...)
    - Dimension: 2560 ✓
    - Type: float ✓
    - Sample values: [-0.00027407, 0.01592491, -0.00934723, ...]
    - Has INT8: False ✓
  ...

Tier1 Summary:
  - Items with embeddings: 92/100
  - Correct dimension (2560): 92/92
  - Correct type (float32): 92/92

================================================================================
OVERALL SUMMARY
================================================================================
Total items: 200
Items with embeddings: 177/200 (88.5%)
Correct dimension (2560): 177/177 (100.0%)
Correct type (float32): 177/177 (100.0%)

✓ Embedding coverage: 88.5% (target: ≥50%)
================================================================================
✓ QWEN4B COREML TEST PASSED
================================================================================
```

---

## Impact on SDK

The SDK (`papr-pythonSDK`) will now receive embeddings for more items:

1. **Faster initialization**: Fewer local embeddings to generate
2. **Better CoreML/ANE utilization**: Server provides float32 embeddings ready for fp16 conversion
3. **Consistent dimensions**: All embeddings are 2560-dim (Qwen4B standard)
4. **Reduced ANE load**: Less time spent on local generation = more ANE headroom for search queries

---

## Environment Variables

The SDK should use these settings for optimal performance:

```bash
# Enable server-provided embeddings
PAPR_INCLUDE_SERVER_EMBEDDINGS=true
PAPR_EMBED_LIMIT=200
PAPR_EMBED_MODEL=Qwen4B

# For CoreML/ANE, request float32 format
PAPR_EMBEDDING_FORMAT=float32  # Auto-set when CoreML enabled
```

---

## Migration Notes

**No breaking changes!** This is a pure enhancement:

- Old SDK versions: Will work as before (generate all embeddings locally)
- New SDK versions: Will benefit from server-provided embeddings automatically
- Server backward compatible: `include_embeddings=false` works as before

---

## Future Improvements

1. **Qdrant indexing coverage**: Ensure all memories get indexed in Qdrant
2. **Embedding cache warming**: Pre-compute embeddings for frequently accessed memories
3. **Smart embed_limit**: Server could prioritize recent/popular items for embedding enrichment
4. **Batch size optimization**: Tune Qdrant batch retrieval for better performance

---

## Files Changed

1. **`routers/v1/sync_routes.py`** - Enable tier0 embeddings, improve tier1 logging
2. **`tests/test_sync_v1.py`** - Add comprehensive Qwen4B/CoreML test case
3. **`models/parse_server.py`** - Add explicit `embedding` and `embedding_int8` fields to `Memory` model

### Memory Model Updates

Added two optional fields to the `Memory` Pydantic model:

```python
# Embedding fields (optional, populated when include_embeddings=True in sync_tiers)
embedding: Optional[List[float]] = Field(
    default=None,
    description="Full precision (float32) embedding vector from Qdrant. Typically 2560 dimensions for Qwen4B. Used for CoreML/ANE fp16 models."
)
embedding_int8: Optional[List[int]] = Field(
    default=None,
    description="Quantized INT8 embedding vector (values -128 to 127). 4x smaller than float32. Default format for efficiency."
)
```

These fields:
- Are **optional** (default=None) - only populated when `include_embeddings=True`
- Support both formats:
  - `embedding`: float32 for CoreML/ANE (full precision, 2560 dims for Qwen4B)
  - `embedding_int8`: quantized INT8 (4x smaller, efficient transfer)
- Are properly typed for validation and IDE autocomplete
- Automatically serialized in JSON responses

