# Qdrant Collection Name Selection Fix

## Problem

The Qdrant collection selection logic had a critical bug where it **always preferred the `Qwen4B` collection** regardless of which embedding model was configured. This caused dimension mismatches when using the 0.6B model (1024 dims) with a collection expecting 2560 dims.

## Root Cause

### Bug 1: init_qdrant_collections_opensource.py (Line 103)
```python
# WRONG: Always used QDRANT_COLLECTION_QWEN4B regardless of dimensions
main_collection = main_collection or os.getenv("QDRANT_COLLECTION_QWEN4B", "Qwen4B")
```

### Bug 2: memory_graph.py (Line 689)
```python
# WRONG: Always preferred Qwen4B first, ignoring LOCAL_EMBEDDING_DIMENSIONS
if env.get("QDRANT_COLLECTION_QWEN4B"):
    self.qdrant_collection = env.get("QDRANT_COLLECTION_QWEN4B")
elif env.get("QDRANT_COLLECTION_QWEN0pt6B"):
    self.qdrant_collection = env.get("QDRANT_COLLECTION_QWEN0pt6B")
```

## What Happened During Your Test

1. `.env` had `LOCAL_EMBEDDING_DIMENSIONS=1024` (0.6B model)
2. Init script created `Qwen4B` collection with **1024 dimensions** ✅
3. But there was an **old** `Qwen4B` collection with **2560 dimensions** still present
4. memory_graph.py selected `Qwen4B` but tried to insert 1024-dim vectors
5. **Error:** `Vector dimension error: expected dim: 2560, got 1024`

## The Fix

### Fixed: init_qdrant_collections_opensource.py
```python
# Auto-select collection based on dimensions
if main_collection is None:
    if embedding_dimensions == 1024:
        main_collection = os.getenv("QDRANT_COLLECTION_QWEN0pt6B", "Qwen0pt6B")
    else:
        main_collection = os.getenv("QDRANT_COLLECTION_QWEN4B", "Qwen4B")
```

### Fixed: memory_graph.py (Constructor)
```python
# Auto-select based on LOCAL_EMBEDDING_DIMENSIONS
embedding_dimensions = int(env.get("LOCAL_EMBEDDING_DIMENSIONS", "1024"))
if embedding_dimensions == 1024:
    self.qdrant_collection = env.get("QDRANT_COLLECTION_QWEN0pt6B", "Qwen0pt6B")
    logger.info(f"Qdrant collection set to: {self.qdrant_collection} (1024 dimensions)")
elif embedding_dimensions == 2560:
    self.qdrant_collection = env.get("QDRANT_COLLECTION_QWEN4B", "Qwen4B")
    logger.info(f"Qdrant collection set to: {self.qdrant_collection} (2560 dimensions)")
```

### Fixed: memory_graph.py (init_qdrant)
Same auto-selection logic applied to ensure consistency.

## Migration Guide

### For Users with 0.6B Model (1024 dims)

Clean restart to use the correct collection name:

```bash
# Delete the misnamed Qwen4B collection
curl -X DELETE http://localhost:6333/collections/Qwen4B

# Rebuild and restart
docker compose down
docker compose build --no-cache
docker compose up -d

# Verify correct collection was created
curl http://localhost:6333/collections | jq
# Should show "Qwen0pt6B" with 1024 dimensions
```

### For Users with 4B Model (2560 dims)

Update your `.env`:
```bash
LOCAL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
LOCAL_EMBEDDING_DIMENSIONS=2560
```

Then restart:
```bash
docker compose restart
```

## Benefits

1. ✅ **Auto-Detection**: System automatically selects correct collection based on dimensions
2. ✅ **Correct Naming**: Qwen0pt6B for 1024 dims, Qwen4B for 2560 dims
3. ✅ **No Confusion**: Collection names now match the model being used
4. ✅ **Backward Compatible**: Fallback logic handles edge cases
5. ✅ **Clear Logs**: Dimension information included in log messages

## Verification

After rebuild, check logs:
```bash
docker logs papr-memory 2>&1 | grep "Qdrant collection"
```

Should see:
```
Qdrant collection set to: Qwen0pt6B (1024 dimensions)
```

And verify collection:
```bash
curl http://localhost:6333/collections/Qwen0pt6B | jq '.result.config.params.vectors'
```

Should show:
```json
{
  "size": 1024,
  "distance": "Cosine",
  ...
}
```

## Files Modified

1. `scripts/opensource/init_qdrant_collections_opensource.py` - Fixed collection name selection
2. `memory/memory_graph.py` - Fixed collection selection in constructor and init_qdrant()
