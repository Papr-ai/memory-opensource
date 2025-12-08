# TensorLake Content Extraction Fix

## Problem Identified

The TensorLake provider was storing reference IDs instead of actual parsed content:

```json
{
  "content": "{\n  \"file_id\": \"file_8mrB8zRQknwmFQpCTrN6C\",\n  \"parse_id\": \"parse_r6QCjC6j6MwwtTKFfdfbz\"\n}"
}
```

This caused downstream activities to create memories with IDs instead of actual document text.

## Root Cause

Looking at the Temporal workflow events (event ID 31, line 983):
```json
"content": "{\n  \"file_id\": \"file_8mrB8zRQknwmFQpCTrN6C\",\n  \"parse_id\": \"parse_r6QCjC6j6MwwtTKFfdfbz\"\n}"
```

The issue was in `core/document_processing/providers/tensorlake.py`:
- The `wait_for_completion()` method might not return the full parsed result with chunks
- The provider was falling back to: `f"Document processed (parse_id: {parse_id})"` when no chunks were found

## Fix Applied

### Updated: `core/document_processing/providers/tensorlake.py`

**Before:**
```python
# Step 4: Wait for completion using SDK
result = await asyncio.to_thread(doc_ai.wait_for_completion, parse_id)

# Step 5: Check status and extract content from chunks
if result.status == ParseStatus.SUCCESSFUL:
    content_parts = []
    if result.chunks:
        # Extract content...
```

**After:**
```python
# Step 4: Wait for completion using SDK
result = await asyncio.to_thread(doc_ai.wait_for_completion, parse_id)

# Step 5: Get the full parsed result with all chunks
# wait_for_completion might not return all data, so fetch explicitly
logger.info(f"Fetching full parse result with chunks...")
result = await asyncio.to_thread(doc_ai.get_parsed_result, parse_id)
logger.info(f"Fetched result: status={result.status}, chunks={len(result.chunks) if result.chunks else 0}")

# Step 6: Check status and extract content from chunks
if result.status == ParseStatus.SUCCESSFUL or result.status == ParseStatus.COMPLETED:
    content_parts = []
    if result.chunks:
        logger.info(f"Extracting content from {len(result.chunks)} chunks")
        for chunk in result.chunks:
            if hasattr(chunk, 'content') and chunk.content:
                content_parts.append(chunk.content)
                logger.debug(f"Chunk content length: {len(chunk.content)}")
    else:
        logger.warning(f"No chunks found in result! Result type: {type(result)}")
```

### Key Changes:

1. **Explicit Fetch**: Added `doc_ai.get_parsed_result(parse_id)` after waiting for completion
2. **Better Status Check**: Added `or result.status == ParseStatus.COMPLETED` as fallback
3. **Enhanced Logging**: Added detailed logging to track content extraction
4. **Error Detection**: Log CRITICAL errors if no content is extracted

## Verification

### Unit Test (Standalone)

✅ **PASSED**: `tests/test_tensorlake_sdk_simple.py`
```bash
poetry run python tests/test_tensorlake_sdk_simple.py
```

Result:
```
✅ Successfully extracted 1399 chars from document
📄 FIRST PAGE:
   Content length: 1399 chars
   First 500 chars:

PROCESS OVERVIEW
Nuance Voice ID: Quick Reference Guide
Answering Calls Using the Voice ID System...
```

### End-to-End Test (Next Step)

Need to run full workflow test to verify:
```bash
poetry run pytest tests/test_document_processing_v2.py::test_document_upload_v2_with_tensorlake_provider -v -s
```

## Expected Flow

### 1. `process_document_with_provider_from_reference`

**Input**: File reference (URL, filename, size)
**Action**: 
- Download file
- Call TensorLake SDK to process
- Extract content from `result.chunks[].content`
- Store in Parse Post via `create_post_with_provider_json`

**Output**:
```json
{
  "post": {"objectId": "wtA0rFElIk"},
  "preview": "PROCESS OVERVIEW\nNuance Voice ID...",  // ← ACTUAL CONTENT
  "stats": {
    "total_pages": 1,
    "provider": "tensorlake"
  }
}
```

### 2. Parse Storage (`create_post_with_provider_json`)

**Stores**:
```json
{
  "content": {
    "provider": "tensorlake",
    "provider_result_file": {  // ← Gzipped JSON file
      "__type": "File",
      "name": "provider_result_4d262c93....json",
      "url": "https://..."
    }
  }
}
```

**provider_result_file contains**:
```json
{
  "file_id": "file_8mrB8zRQknwmFQpCTrN6C",
  "parse_id": "parse_r6QCjC6j6MwwtTKFfdfbz",
  "content": "PROCESS OVERVIEW\nNuance Voice ID...",  // ← ACTUAL CONTENT!
  "status": "successful",
  "parsed_pages_count": 1,
  "chunks_count": 1
}
```

### 3. `extract_structured_content_from_provider`

**Input**: `post_id` (wtA0rFElIk)
**Action**:
- Fetch Post from Parse
- Download `provider_result_file` (gzipped)
- Decompress and parse JSON
- Extract `content` field

**Output**:
```json
{
  "decision": "simple",
  "memory_requests": [{
    "content": "PROCESS OVERVIEW\nNuance Voice ID...",  // ← ACTUAL CONTENT!
    "type": "document",
    "metadata": {...}
  }]
}
```

### 4. `store_batch_memories_in_parse_for_processing`

**Input**: Memory requests with actual content
**Action**: Create BatchMemoryRequest Post with content
**Result**: Batch processing workflow creates Memory objects with actual text

## What Changed vs. Before

### ❌ Before (Broken)

```
TensorLake result.chunks = [] (empty)
↓
Fallback: content = f"Document processed (parse_id: {parse_id})"
↓
Post stores: {"file_id": "...", "parse_id": "..."}
↓
Memory created with: "{\n  \"file_id\": \"...\",\n  \"parse_id\": \"...\"\n}"
```

### ✅ After (Fixed)

```
TensorLake wait_for_completion()
↓
Explicit get_parsed_result(parse_id) ← NEW!
↓
result.chunks = [Chunk(content="PROCESS OVERVIEW...")]
↓
content = "PROCESS OVERVIEW\nNuance Voice ID..."
↓
Post stores: {
  "file_id": "...",
  "parse_id": "...",
  "content": "PROCESS OVERVIEW..."  ← ACTUAL TEXT!
}
↓
Memory created with actual document content
```

## Testing Checklist

- [x] Unit test passes (`tests/test_tensorlake_sdk_simple.py`)
- [ ] E2E test passes (`test_document_upload_v2_with_tensorlake_provider`)
- [ ] Verify Memory objects contain actual text (not IDs)
- [ ] Check Parse Post contains `provider_specific.content` field
- [ ] Confirm batch memory workflow processes actual content

## Other Providers

Created unit tests for other providers to verify they also extract content correctly:

1. **Gemini**: `tests/test_gemini_provider_simple.py`
2. **PaddleOCR**: `tests/test_paddleocr_provider_simple.py`
3. **DeepSeek-OCR**: `tests/test_deepseek_ocr_provider_simple.py`

See `docs/PROVIDER_UNIT_TESTS_README.md` for running instructions.

## SDK Versions

Current (from `pyproject.toml`):
- `tensorlake = "*"` ← Using latest
- `google-generativeai = "^0.8.0"`
- `httpx = "^0.28.1"`
- `certifi = "^2024.8.30"`
- `paddleocr` - Not installed yet

To update:
```bash
poetry update tensorlake google-generativeai
```

## Next Steps

1. **Restart Temporal Workers** (to pick up new code):
   ```bash
   pkill -f "start_temporal_worker.py\|start_document_worker.py"
   cd /Users/shawkatkabbara/Documents/GitHub/memory && \
   poetry run python start_temporal_worker.py > .temporal_worker.out 2>&1 & \
   poetry run python start_document_worker.py > .document_worker.out 2>&1 &
   ```

2. **Run E2E Test**:
   ```bash
   poetry run pytest tests/test_document_processing_v2.py::test_document_upload_v2_with_tensorlake_provider -v -s
   ```

3. **Check Memory Content**:
   - Query Memory object created by workflow
   - Verify `content` field has actual text, not just IDs

4. **Test Other Providers**:
   - Run unit tests for Gemini, PaddleOCR, DeepSeek-OCR
   - Add E2E tests for each provider

