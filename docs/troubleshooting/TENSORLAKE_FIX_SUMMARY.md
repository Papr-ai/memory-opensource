# TensorLake Integration Fix Summary

## Problem

The TensorLake provider is currently storing `file_id` and `parse_id` references as content instead of fetching the actual parsed text:

```json
{
  "content": "{\"file_id\": \"file_8mrB8zRQknwmFQpCTrN6C\", \"parse_id\": \"parse_zRh8FKgWm86ddk9NdTbnf\"}"
}
```

This results in memories containing references rather than searchable text content.

## Root Cause

The workflow has two places where this needs to be fixed:

### 1. TensorLake Provider (`core/document_processing/providers/tensorlake.py`)

**Current behavior**: Uses HTTP API but doesn't properly extract content from the response

**Should do**: Use TensorLake SDK to get the parsed content:
```python
from tensorlake.documentai import DocumentAI

doc_ai = DocumentAI(api_key=self.api_key)
result = doc_ai.get_parse_result(parse_id=parse_id)

# Result has chunks with content
full_text = "\n".join(chunk.content for chunk in result.chunks)
```

### 2. Extract Activity (`cloud_plugins/temporal/activities/document_activities.py`)

**Lines 1197-1237**: The dereferencing logic fetches content but the structure isn't being handled correctly.

**Current code** (simplified):
```python
if parse_id and not provider_specific.get("content"):
    parse_result = await tensorlake_provider.fetch_parse_result(parse_id)
    content = parse_result.get("content") or parse_result.get("text", "")
    provider_specific = {
        "file_id": file_id,
        "parse_id": parse_id,
        "content": content,  # This is set
        ...
    }
```

**Issue**: The `fetch_parse_result` method returns HTTP response JSON, not SDK result object. The response may not have a simple `content` field - it likely has `chunks` structure from the SDK.

## Solution

### Step 1: Install TensorLake SDK
```bash
pip install tensorlake
# Or add to pyproject.toml
poetry add tensorlake
```

### Step 2: Update TensorLake Provider

Modify `core/document_processing/providers/tensorlake.py` to use the SDK:

```python
async def fetch_parse_result(self, parse_id: str) -> Dict[str, Any]:
    """Fetch the full parsed result using TensorLake SDK"""
    logger.info(f"Fetching TensorLake parse result for parse_id: {parse_id}")
    
    try:
        from tensorlake.documentai import DocumentAI
        
        # Initialize SDK client
        doc_ai = DocumentAI(api_key=self.api_key)
        
        # Get parse result using SDK
        result = doc_ai.get_parse_result(parse_id=parse_id)
        
        # Extract content from chunks
        if hasattr(result, 'chunks') and result.chunks:
            content_parts = []
            for chunk in result.chunks:
                if hasattr(chunk, 'content'):
                    content_parts.append(chunk.content)
            
            full_content = "\n".join(content_parts)
            
            # Return structured result
            return {
                "parse_id": parse_id,
                "status": result.status if hasattr(result, 'status') else "completed",
                "content": full_content,
                "chunks": [{"content": chunk.content} for chunk in result.chunks if hasattr(chunk, 'content')],
                "metadata": getattr(result, 'metadata', {})
            }
        else:
            raise ValueError(f"TensorLake result for {parse_id} has no chunks")
            
    except Exception as e:
        logger.error(f"Failed to fetch TensorLake parse result: {e}")
        raise Exception(f"Failed to fetch TensorLake result: {str(e)}")
```

### Step 3: Update Provider Adapter

Ensure `provider_to_markdown` in `core/document_processing/provider_adapter.py` handles the new structure:

```python
def provider_to_markdown(provider_name: str, provider_specific: Dict[str, Any]) -> str:
    name = provider_name.lower()
    
    if name == "tensorlake":
        # Priority 1: Direct content field (from fetch_parse_result)
        content = provider_specific.get("content")
        if content:
            logger.info(f"TensorLake: using content field ({len(content)} chars)")
            return content
        
        # Priority 2: Combine chunks
        chunks = provider_specific.get("chunks", [])
        if chunks:
            text_parts = []
            for chunk in chunks:
                if isinstance(chunk, dict) and "content" in chunk:
                    text_parts.append(chunk["content"])
            if text_parts:
                combined = "\n".join(text_parts)
                logger.info(f"TensorLake: combined {len(chunks)} chunks ({len(combined)} chars)")
                return combined
        
        # Priority 3: Check full_result (from dereferencing)
        full_result = provider_specific.get("full_result", {})
        if full_result:
            content = full_result.get("content")
            if content:
                logger.info(f"TensorLake: using full_result.content ({len(content)} chars)")
                return content
        
        # Error: should not reach here if dereferencing worked
        parse_id = provider_specific.get("parse_id")
        if parse_id:
            logger.error(f"TensorLake: no content found, only parse_id reference: {parse_id}")
            raise ValueError(f"TensorLake result has no content, only parse_id: {parse_id}")
        
        # Fallback to JSON dump
        return json.dumps(provider_specific, indent=2)
```

### Step 4: Test the Fix

Run the test script to verify SDK output:
```bash
export TENSORLAKE_API_KEY="your_key"
poetry run python test_provider_outputs.py
```

Expected output:
```
📊 PARSE RESULT STRUCTURE (SDK):
✅ Has 'chunks' attribute
📦 First chunk structure:
✅ Chunk has 'content' attribute
   Length: 5432 chars
   First 500 chars:
   [actual document text here]
```

Then run the integration test:
```bash
poetry run pytest tests/test_document_processing_v2.py::test_document_upload_v2_with_tensorlake_provider -v
```

## Expected Behavior After Fix

1. TensorLake provider uploads document and starts parsing
2. Provider polls for completion using SDK
3. When complete, SDK returns `result.chunks` with actual content
4. Provider extracts text from chunks and returns in `provider_specific`
5. Temporal activity detects the proper structure
6. `provider_to_markdown` extracts the content
7. Memories are created with actual searchable text

## Verification

After the fix, check that:
- [ ] Test script shows SDK returns `result.chunks` with `chunk.content`
- [ ] `fetch_parse_result` properly extracts content from SDK response
- [ ] `provider_to_markdown` successfully extracts text
- [ ] Integration test passes
- [ ] Created memories contain actual document text, not references

## Files to Modify

1. ✅ `test_provider_outputs.py` - Updated to use SDK
2. ⏳ `core/document_processing/providers/tensorlake.py` - Update `fetch_parse_result` to use SDK
3. ⏳ `core/document_processing/provider_adapter.py` - Update `provider_to_markdown` for proper extraction
4. ⏳ `pyproject.toml` - Add `tensorlake` dependency

## Next Steps

1. **Run test script**: `poetry run python test_provider_outputs.py`
2. **Verify SDK output**: Confirm it returns `result.chunks` with `chunk.content`
3. **Update provider**: Implement SDK-based `fetch_parse_result`
4. **Update adapter**: Ensure proper content extraction
5. **Test integration**: Run pytest to verify e2e flow
6. **Deploy**: Once tests pass, deploy the fix

