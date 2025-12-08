# Provider Testing Guide

## Overview

This guide explains how to test document processing with different providers and how to ensure the parsed response from each provider is properly passed through the Temporal workflow pipeline.

## Supported Providers

We now support the following document processing providers:

1. **Reducto** - Advanced structured document parsing with table/image extraction
2. **Gemini** - Google's multimodal AI for document understanding
3. **TensorLake** - Specialized document processing service
4. **PaddleOCR** - Open-source OCR solution from Baidu
5. **DeepSeek-OCR** - Deep learning based OCR

## Provider Enum

All providers are defined in `models/shared_types.py`:

```python
class PreferredProvider(str, Enum):
    GEMINI = "gemini"
    TENSORLAKE = "tensorlake"
    REDUCTO = "reducto"
    PADDLEOCR = "paddleocr"
    DEEPSEEK_OCR = "deepseek-ocr"
    AUTO = "auto"
```

## Test Cases

### Provider-Specific Tests

Each provider has a dedicated end-to-end test in `tests/test_document_processing_v2.py`:

1. **`test_document_upload_v2_with_gemini_provider`**
   - Tests Gemini provider
   - Requires: `GOOGLE_API_KEY` or `GEMINI_API_KEY`
   - Verifies: Document upload, processing, and memory creation

2. **`test_document_upload_v2_with_tensorlake_provider`**
   - Tests TensorLake provider
   - Requires: `TENSORLAKE_API_KEY`
   - Verifies: Document upload, processing, and memory creation

3. **`test_document_upload_v2_with_paddleocr_provider`**
   - Tests PaddleOCR provider
   - Requires: PaddleOCR library installed
   - Verifies: Document upload, processing, and memory creation

4. **`test_document_upload_v2_with_deepseek_ocr_provider`**
   - Tests DeepSeek-OCR provider
   - Requires: `DEEPSEEK_API_KEY`
   - Verifies: Document upload, processing, and memory creation

5. **`test_document_upload_v2_with_real_pdf_file`**
   - Tests Reducto provider (existing test)
   - Requires: `REDUCTO_API_KEY`
   - Verifies: Full end-to-end flow with hierarchical chunking

## How Provider Adapters Work

### 1. Provider Response Flow

```
Provider API Response 
  ↓
Provider-Specific SDK Types (if available)
  ↓
Provider Adapter (core/document_processing/provider_adapter.py)
  ↓
ContentElements (unified format)
  ↓
Temporal Workflow
  ↓
LLM Memory Generation
  ↓
Memory Creation
```

### 2. Provider Adapter Implementation

The provider adapter (`core/document_processing/provider_adapter.py`) handles converting each provider's response format into our unified `ContentElement` format:

#### Reducto Adapter
- Uses Reducto SDK's `PipelineResponse` type for type-safe parsing
- Extracts `result.parse.result.chunks[].blocks[]`
- Identifies block types: `text`, `table`, `image`, `figure`
- Creates appropriate `ContentElement` subtypes:
  - `TextElement` for text/paragraph/heading
  - `TableElement` for tables
  - `ImageElement` for images/figures with bbox coordinates

#### Gemini Adapter
- Parses Google AI SDK response format
- Extracts text content from candidates
- Creates `TextElement` objects

#### TensorLake Adapter
- Uses `ProviderContentExtractor.extract_from_tensorlake()`
- Extracts pages and content structure
- Creates appropriate `ContentElement` types

#### PaddleOCR Adapter
- Parses OCR results with bounding boxes
- Handles table detection via metadata
- Creates `TextElement` and `TableElement` objects

#### DeepSeek-OCR Adapter
- Parses OCR results by page
- Extracts full text per page
- Creates `TextElement` objects

### 3. Temporal Workflow Integration

The document processing workflow (`cloud_plugins/temporal/workflows/document_processing.py`) handles provider responses through these steps:

1. **Upload & Validation** - `download_and_validate_file` activity
2. **Provider Processing** - `process_document_with_provider_from_reference` activity
   - Calls provider's `process_document()` method
   - Stores provider JSON in Parse Server Post
   - Returns minimal payload with Post ID reference

3. **Structured Extraction** - `extract_structured_content_from_provider` activity
   - Fetches provider JSON from Parse Server
   - Uses provider adapter to extract `ContentElement[]`
   - Handles both simple (markdown) and complex (structured) paths

4. **LLM Generation** - `generate_llm_optimized_memory_structures` activity
   - Processes `ContentElement[]` into optimized memory structures
   - Uses LLM for complex content (tables, images)
   - Creates `AddMemoryRequest[]` objects

5. **Memory Creation** - Batch memory workflow
   - Stores memories in Parse Server
   - Triggers full indexing pipeline
   - Links memories to document Post

### 4. Provider-Specific Handling

Each provider adapter ensures:

#### Type Safety
- Uses provider SDKs when available (e.g., Reducto's `PipelineResponse`)
- Falls back to dict navigation if SDK parsing fails
- Validates response structure

#### Content Preservation
- Never truncates or summarizes content
- Preserves all metadata (page numbers, confidence scores, etc.)
- Maintains hierarchical relationships

#### Multi-Modal Support
- Extracts tables with structured data
- Handles images with URLs and descriptions
- Preserves charts and diagrams

## Running the Tests

### Run All Provider Tests
```bash
poetry run pytest tests/test_document_processing_v2.py::test_document_upload_v2_with_gemini_provider -xvs
poetry run pytest tests/test_document_processing_v2.py::test_document_upload_v2_with_tensorlake_provider -xvs
poetry run pytest tests/test_document_processing_v2.py::test_document_upload_v2_with_paddleocr_provider -xvs
poetry run pytest tests/test_document_processing_v2.py::test_document_upload_v2_with_deepseek_ocr_provider -xvs
poetry run pytest tests/test_document_processing_v2.py::test_document_upload_v2_with_real_pdf_file -xvs
```

### Run with Specific Provider
```bash
# Test Gemini
TEST_X_USER_API_KEY=your_key GOOGLE_API_KEY=your_gemini_key poetry run pytest tests/test_document_processing_v2.py::test_document_upload_v2_with_gemini_provider -xvs

# Test TensorLake
TEST_X_USER_API_KEY=your_key TENSORLAKE_API_KEY=your_key poetry run pytest tests/test_document_processing_v2.py::test_document_upload_v2_with_tensorlake_provider -xvs

# Test Reducto
TEST_X_USER_API_KEY=your_key REDUCTO_API_KEY=your_key poetry run pytest tests/test_document_processing_v2.py::test_document_upload_v2_with_real_pdf_file -xvs
```

## Verifying Provider Integration

### What Each Test Verifies

1. **Upload Success**
   - HTTP 200/202 response
   - Valid upload_id returned
   - Document status tracking initiated

2. **Provider Processing**
   - Provider successfully processes PDF
   - Provider-specific response format is captured
   - Response is stored in Parse Server

3. **Adapter Conversion**
   - Provider response is converted to ContentElements
   - All content is preserved (no truncation)
   - Metadata is properly mapped

4. **Temporal Workflow**
   - Workflow progresses through all stages
   - No payload size limit errors
   - Status updates are tracked

5. **Memory Creation**
   - Memories are created successfully
   - Memories link to document Post
   - Hierarchical structure is preserved

### Common Issues and Solutions

#### Issue: Provider API Key Missing
**Solution**: Set the required environment variable:
- Gemini: `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- TensorLake: `TENSORLAKE_API_KEY`
- Reducto: `REDUCTO_API_KEY`
- DeepSeek: `DEEPSEEK_API_KEY`

#### Issue: Payload Size Limit Error
**Solution**: Already handled! Large provider responses are:
1. Stored in Parse Server Post
2. Referenced by Post ID in Temporal workflow
3. Fetched on-demand in activities

#### Issue: Provider Response Format Changed
**Solution**: 
1. Check provider adapter in `core/document_processing/provider_adapter.py`
2. Update SDK type if using typed parsing
3. Add fallback dict navigation

#### Issue: ContentElement Conversion Failed
**Solution**:
1. Check logs for specific provider adapter error
2. Verify provider response structure matches expected format
3. Add provider-specific handling if needed

## Adding New Providers

To add a new provider:

1. **Create Provider Class** (`core/document_processing/providers/`)
   - Extend `DocumentProvider` base class
   - Implement `process_document()` method
   - Return `DocumentProcessingResult`

2. **Register Provider** (`core/document_processing/provider_manager.py`)
   ```python
   _providers = {
       "newprovider": NewProviderClass
   }
   ```

3. **Add to Enum** (`models/shared_types.py`)
   ```python
   class PreferredProvider(str, Enum):
       NEWPROVIDER = "newprovider"
   ```

4. **Create Adapter** (`core/document_processing/provider_adapter.py`)
   ```python
   elif name == "newprovider":
       return _extract_newprovider_elements(...)
   ```

5. **Add Test** (`tests/test_document_processing_v2.py`)
   - Create `test_document_upload_v2_with_newprovider_provider()`
   - Follow existing test patterns
   - Verify end-to-end flow

## Best Practices

1. **Always Use Hierarchical Mode**
   - Set `hierarchical_enabled=True` for best results
   - Enables structured extraction and LLM optimization

2. **Test with Real Documents**
   - Use actual PDFs, not synthetic test files
   - Verify with documents containing tables/images

3. **Check Provider Response**
   - Inspect Parse Server Post's `provider_specific` field
   - Verify all expected content is captured

4. **Monitor Temporal Workflow**
   - Check Temporal UI for workflow execution
   - Review activity results for each stage

5. **Verify Memory Quality**
   - Query Parse Server Memory collection
   - Check Neo4j for proper graph structure
   - Verify Qdrant embeddings are created

## Debugging Tips

### View Provider Response
```python
# In Parse Server dashboard:
# 1. Open Post collection
# 2. Find by upload_id
# 3. View provider_specific field
```

### Check Temporal Workflow
```bash
# View workflow history
temporal workflow show --workflow-id "document-processing-{upload_id}"

# View activity results
temporal workflow show --workflow-id "document-processing-{upload_id}" --show-detail
```

### Inspect ContentElements
```python
# Add logging in extract_structured_content_from_provider activity
logger.info(f"Extracted elements: {[e.content_type for e in structured_elements]}")
```

## Conclusion

With these tests in place, we ensure that:
1. ✅ All providers work correctly
2. ✅ Provider responses are properly adapted
3. ✅ Temporal workflows handle all provider formats
4. ✅ Memories are created successfully
5. ✅ No data is lost in the pipeline

The provider adapter layer is the key component that ensures consistent behavior across all providers!

