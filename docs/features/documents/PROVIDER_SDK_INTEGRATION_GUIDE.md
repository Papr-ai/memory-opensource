# Provider SDK Integration Guide

## Overview

This guide documents how to properly integrate document processing provider SDKs (Gemini, TensorLake, PaddleOCR, DeepSeek-OCR) into our workflow. The key is understanding what data structures each SDK returns and creating appropriate Pydantic models.

## Key Findings from Research

### Gemini Vision API

**Structured Output Support**: ✅ YES

Gemini supports structured output through prompts with JSON schemas. You can:

1. **Prompt-based approach**: Specify desired JSON format in the prompt
2. **response_mime_type parameter**: Use `generation_config={"response_mime_type": "application/json"}`
3. **Pydantic model validation**: Define Pydantic models for the expected structure

**Example**:
```python
import google.generativeai as genai

# Define Pydantic model for structured output
from pydantic import BaseModel
from typing import List

class DocumentStructure(BaseModel):
    title: str
    author: str
    content: str
    sections: List[str]

# In prompt, specify the schema
prompt = """
Extract the following fields from the document:
- title: Document title
- author: Author's name  
- content: Full text content
- sections: List of section headings

Return as JSON matching this schema.
"""

response = model.generate_content(
    [prompt, image_data],
    generation_config={"response_mime_type": "application/json"}
)

# Parse and validate with Pydantic
data = json.loads(response.text)
structured_data = DocumentStructure(**data)
```

**Response Structure**:
- `response.text`: Contains the generated text/JSON
- `response.candidates`: List of candidate responses
- Each candidate has structured content with parts

### PaddleOCR

**Output Format**: List of lists containing `[bbox, (text, confidence)]`

PaddleOCR returns OCR results as nested lists:
```python
[
    [  # Page 1
        [
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],  # BBox coordinates
            ("detected text", 0.98)  # (text, confidence)
        ],
        # ... more detections
    ],
    [  # Page 2 (if multi-page)
        # ...
    ]
]
```

**Pydantic Model**:
```python
from pydantic import BaseModel
from typing import List, Tuple

class PaddleOCRBBox(BaseModel):
    coordinates: List[List[float]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

class PaddleOCRDetection(BaseModel):
    bbox: List[List[float]]
    text: str
    confidence: float

class PaddleOCRResult(BaseModel):
    pages: List[List[PaddleOCRDetection]]
```

### DeepSeek-OCR (HuggingFace API)

**Output Format**: Dictionary or list with `generated_text` or `text` field

HuggingFace Inference API typically returns:
```python
[
    {
        "generated_text": "Full OCR text here...",
        "confidence": 0.95  # optional
    }
]
```

Or sometimes:
```python
{
    "text": "Full OCR text here...",
    "metadata": {...}
}
```

**Pydantic Model**:
```python
class DeepSeekOCRResponse(BaseModel):
    generated_text: Optional[str] = None
    text: Optional[str] = None
    confidence: Optional[float] = None
    
    def get_text(self) -> str:
        """Get text from either field"""
        return self.generated_text or self.text or ""
```

### TensorLake

**SDK Required**: ✅ YES - Use `tensorlake` Python SDK

TensorLake has an official Python SDK that should be used instead of raw HTTP requests:

```bash
pip install tensorlake
```

**SDK Usage**:
```python
from tensorlake.documentai import DocumentAI

# Initialize client
doc_ai = DocumentAI(api_key='your-api-key')

# Parse document
with open('document.pdf', 'rb') as f:
    parse_response = doc_ai.parse_document(file=f, filename='document.pdf')

parse_id = parse_response.parse_id

# Get result (poll until status is 'successful')
result = doc_ai.get_parse_result(parse_id=parse_id)

if result.status == 'successful':
    # Result has chunks attribute
    for chunk in result.chunks:
        print(chunk.content)  # Each chunk has content attribute
```

**SDK Response Structure**:
- `result.status`: Status string ('successful', 'failed', 'processing')
- `result.chunks`: List of chunk objects
- Each `chunk` has:
  - `chunk.content`: The parsed text content
  - `chunk.metadata`: Optional metadata

**Pydantic Model**:
```python
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class TensorLakeChunk(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

class TensorLakeParseResult(BaseModel):
    parse_id: str
    status: str
    chunks: List[TensorLakeChunk]
    
    def get_full_text(self) -> str:
        """Combine all chunks into full text"""
        return "\n".join(chunk.content for chunk in self.chunks)
```

**Current Issue**: The HTTP-based provider stores `file_id` and `parse_id` references, but doesn't fetch the actual content using the SDK's `get_parse_result` method. Need to migrate to SDK-based approach.

## Implementation Strategy

### 1. Run Test Script

First, run the test script to see actual outputs:

```bash
# Set up environment variables
export GOOGLE_API_KEY="your_key"
export TENSORLAKE_API_KEY="your_key"
export HUGGINGFACE_API_KEY="your_key"

# Run test script
poetry run python test_provider_outputs.py
```

### 2. Create Pydantic Models

Based on test results, create typed models in `core/document_processing/provider_models.py`:

```python
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class GeminiResponse(BaseModel):
    """Gemini structured response"""
    title: Optional[str] = None
    content: str
    sections: List[str] = []
    confidence: float = 0.95

class PaddleOCRResponse(BaseModel):
    """PaddleOCR response structure"""
    pages: List[List[Dict[str, Any]]]
    
    def to_text(self) -> str:
        """Convert OCR results to plain text"""
        text_lines = []
        for page in self.pages:
            for detection in page:
                if len(detection) >= 2:
                    text, conf = detection[1]
                    text_lines.append(text)
        return "\n".join(text_lines)

class TensorLakeResponse(BaseModel):
    """TensorLake parse response"""
    parse_id: str
    file_id: str
    status: str
    content: Optional[str] = None
    
    def get_content(self) -> str:
        """Get content, raising error if not available"""
        if not self.content:
            raise ValueError(f"TensorLake result {self.parse_id} has no content")
        return self.content
```

### 3. Update Provider Classes

Modify each provider class to return typed responses:

```python
# In gemini.py
async def process_document(...) -> ProcessingResult:
    # Use structured output
    prompt = """Extract document content as JSON with fields:
    - title: document title
    - content: full text content
    - sections: list of section headings
    """
    
    response = await self._generate_content_async([
        prompt,
        {"mime_type": mime_type, "data": file_content}
    ])
    
    # Parse with Pydantic
    structured_data = GeminiResponse.model_validate_json(response.text)
    
    return ProcessingResult(
        pages=[DocumentPage(
            page_number=1,
            content=structured_data.content,
            confidence=structured_data.confidence
        )],
        provider_specific=structured_data.model_dump()
    )
```

### 4. Fix TensorLake Integration

The issue is in `extract_structured_content_from_provider` - it needs to:

1. Detect `parse_id` reference
2. Call `fetch_parse_result()` to get actual content
3. Extract the `content` field from the response
4. Pass the ACTUAL content to `provider_to_markdown`, not the reference

**Current bug**: The dereferencing happens but the content isn't being extracted properly.

**Fix location**: `cloud_plugins/temporal/activities/document_activities.py` lines 1197-1237

The dereferencing creates:
```python
provider_specific = {
    "file_id": file_id,
    "parse_id": parse_id,
    "content": content,  # ← This is set
    ...
}
```

But then `provider_to_markdown` doesn't extract it properly. Need to verify the `provider_to_markdown` function in `core/document_processing/provider_adapter.py` handles this case.

### 5. Update provider_adapter.py

Ensure each provider has proper content extraction:

```python
def provider_to_markdown(provider_name: str, provider_specific: Dict[str, Any]) -> str:
    name = provider_name.lower()
    
    if name == "gemini":
        # Extract from structured response
        if "content" in provider_specific:
            return provider_specific["content"]
        return provider_specific.get("text", "")
    
    elif name == "tensorlake":
        # Extract from dereferenced result
        content = provider_specific.get("content") or provider_specific.get("text", "")
        if content:
            return content
        # If full_result exists (dereferenced), try that
        full_result = provider_specific.get("full_result", {})
        if full_result:
            content = full_result.get("content") or full_result.get("text", "")
            if content:
                return content
        # Should not reach here if dereferencing worked
        raise ValueError(f"TensorLake result has no content: {list(provider_specific.keys())}")
    
    elif name == "paddleocr":
        # Extract from OCR detections
        if "results" in provider_specific:
            text_lines = []
            for page in provider_specific["results"]:
                text_lines.append(page.get("text", ""))
            return "\n\n".join(text_lines)
    
    elif name == "deepseek-ocr":
        # Extract from HuggingFace response
        if "ocr_results" in provider_specific:
            text_parts = []
            for result in provider_specific["ocr_results"]:
                text_parts.append(result.get("text", ""))
            return "\n\n".join(text_parts)
    
    # Fallback
    return json.dumps(provider_specific, indent=2)
```

## Testing Workflow

1. **Run test script**: `poetry run python test_provider_outputs.py`
2. **Review outputs**: Understand actual data structures
3. **Create Pydantic models**: Based on real outputs
4. **Update provider classes**: Use typed responses
5. **Update provider_adapter**: Proper content extraction
6. **Run integration tests**: Verify e2e flow works

## Common Issues

### Issue: TensorLake returns `file_id`/`parse_id` as content

**Root Cause**: The dereferencing logic fetches the content but it's not being extracted properly by `provider_to_markdown`.

**Solution**: 
1. Verify `fetch_parse_result` returns the actual `content` field
2. Ensure `provider_to_markdown` extracts from the dereferenced structure
3. Add logging to trace where content is lost

### Issue: Gemini returns unstructured text

**Root Cause**: Not using structured output features.

**Solution**: Use `response_mime_type: "application/json"` and Pydantic validation.

### Issue: PaddleOCR/DeepSeek return unexpected format

**Root Cause**: API may return different formats depending on model/version.

**Solution**: Use the test script to verify actual format, then handle multiple possible structures.

## Next Steps

1. ✅ Create test script to verify provider outputs
2. ⏳ Run test script with real API keys
3. ⏳ Create Pydantic models based on actual outputs
4. ⏳ Fix TensorLake content extraction bug
5. ⏳ Update all provider classes to use typed responses
6. ⏳ Update provider_adapter.py with proper extraction logic
7. ⏳ Run integration tests to verify e2e flow

