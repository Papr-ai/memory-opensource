# Running Provider Output Tests

## Quick Start

```bash
# 1. Set up environment variables
export GOOGLE_API_KEY="your_gemini_key"
export TENSORLAKE_API_KEY="your_tensorlake_key"
export HUGGINGFACE_API_KEY="your_huggingface_key"

# 2. Run the test script
poetry run python test_provider_outputs.py
```

## What This Tests

The script will:

1. **Test Gemini Vision API**
   - Upload a PDF/image
   - Test basic text extraction
   - Test structured JSON output
   - Show what fields/structure Gemini returns

2. **Test TensorLake**
   - Upload file
   - Start parsing
   - Poll for completion
   - Show the actual structure of parse results
   - **VERIFY**: Does it have `content` or `text` field?

3. **Test PaddleOCR**
   - Run OCR on test file
   - Show the nested list structure
   - Display bbox + text + confidence format

4. **Test DeepSeek-OCR (HuggingFace)**
   - Call HuggingFace Inference API
   - Show response structure
   - Check for `generated_text`, `text`, etc.

## Expected Outputs

For each provider, you'll see:

```
================================================================================
TESTING: [Provider Name]
================================================================================

📊 RESPONSE STRUCTURE:
Type: <class 'dict'>
Keys: ['content', 'metadata', ...]

📝 RESPONSE TEXT:
Type: <class 'str'>
Length: 5432 chars
First 500 chars:
[actual text content]
```

## Troubleshooting

### Gemini Error: "API key not set"
```bash
export GOOGLE_API_KEY="your_key_here"
# OR
export GEMINI_API_KEY="your_key_here"
```

### TensorLake Error: 401 Unauthorized
```bash
export TENSORLAKE_API_KEY="your_key_here"
```

### Test file not found
The script looks for `tests/fixtures/call_answering_sop.pdf`. Make sure this file exists:
```bash
ls -la tests/fixtures/call_answering_sop.pdf
```

### Package not installed
```bash
# For Gemini
pip install google-generativeai

# For TensorLake (SDK required!)
pip install tensorlake

# For PaddleOCR
pip install paddleocr

# For HTTP clients
pip install httpx
```

## What to Look For

### 1. Gemini
✅ **Good**: Returns clean text in `response.text`
✅ **Good**: Supports structured JSON output with schema
❌ **Bad**: Only returns raw text without structure

### 2. TensorLake
✅ **Good**: Has `content` or `text` field with full document text
❌ **Bad**: Only returns `parse_id` and `file_id` without content

### 3. PaddleOCR
✅ **Good**: Returns list of `[bbox, (text, confidence)]` tuples
✅ **Good**: Confidence scores are reasonable (>0.8)
❌ **Bad**: Returns empty list or null

### 4. DeepSeek-OCR
✅ **Good**: Returns `generated_text` or `text` field
✅ **Good**: Text is clean and well-formatted
❌ **Bad**: Returns error or model loading message

## After Running Tests

1. **Review the output structures** - understand what each provider returns
2. **Check the guide** - see `PROVIDER_SDK_INTEGRATION_GUIDE.md`
3. **Create Pydantic models** - based on actual outputs
4. **Fix the bugs** - especially TensorLake content extraction

## Common Patterns

### Pattern 1: Simple Text Response
```python
{
    "text": "Full document text here...",
    "metadata": {...}
}
```

**Extraction**: `result.get("text")`

### Pattern 2: Structured Response with Multiple Fields
```python
{
    "content": "Main text",
    "title": "Document Title",
    "sections": ["Intro", "Body", "Conclusion"]
}
```

**Extraction**: Use Pydantic model

### Pattern 3: List of Detections
```python
[
    {"text": "Line 1", "confidence": 0.98},
    {"text": "Line 2", "confidence": 0.95}
]
```

**Extraction**: Join all text fields

### Pattern 4: Nested Structure
```python
{
    "result": {
        "parse": {
            "content": "Actual text here"
        }
    }
}
```

**Extraction**: Navigate nested dict

## Next Steps After Testing

1. ✅ Run test script
2. ⏳ Document actual output formats
3. ⏳ Create Pydantic models in `core/document_processing/provider_models.py`
4. ⏳ Update each provider class to use typed responses
5. ⏳ Fix `provider_to_markdown` to extract content properly
6. ⏳ Run integration tests to verify full workflow

## Questions to Answer

- [ ] Does Gemini support structured JSON output?
- [ ] What field does TensorLake use for text content? (`content` vs `text`)
- [ ] Does PaddleOCR return confidence scores?
- [ ] What format does DeepSeek-OCR use? (`generated_text` vs `text`)
- [ ] Do any providers return structured data (tables, sections, etc.)?

Mark each question with your findings!

