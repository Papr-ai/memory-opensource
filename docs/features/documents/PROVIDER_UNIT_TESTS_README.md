# Provider Unit Tests

Simple unit tests to verify each document processing provider extracts content correctly.

## Overview

These tests verify that each provider:
1. ✅ Can initialize with proper SDK/configuration
2. ✅ Processes a test PDF file
3. ✅ Extracts actual text content (not just IDs/references)
4. ✅ Returns content in the expected format

## Test Files

| Provider | Test File | SDK/Package |
|----------|-----------|-------------|
| TensorLake | `tests/test_tensorlake_sdk_simple.py` | `tensorlake` (PyPI) |
| Gemini | `tests/test_gemini_provider_simple.py` | `google-generativeai` |
| PaddleOCR | `tests/test_paddleocr_provider_simple.py` | `paddleocr` |
| DeepSeek-OCR | `tests/test_deepseek_ocr_provider_simple.py` | HTTP API |

## Setup

### 1. Install SDKs

```bash
# TensorLake
poetry add tensorlake

# Gemini
poetry add google-generativeai

# PaddleOCR (local OCR, no API key needed)
poetry add paddleocr

# DeepSeek-OCR (HTTP API, no special SDK)
# Already installed: httpx
```

### 2. Environment Variables

Add to your `.env`:

```bash
# TensorLake
TENSORLAKE_API_KEY=your_tensorlake_api_key

# Gemini
GOOGLE_API_KEY=your_google_api_key
# OR
GEMINI_API_KEY=your_gemini_api_key

# DeepSeek-OCR
DEEPSEEK_API_KEY=your_deepseek_api_key

# PaddleOCR (runs locally, no key needed)
```

## Running Tests

### Run Individual Provider Tests

```bash
# TensorLake
poetry run python tests/test_tensorlake_sdk_simple.py

# Gemini
poetry run python tests/test_gemini_provider_simple.py

# PaddleOCR
poetry run python tests/test_paddleocr_provider_simple.py

# DeepSeek-OCR
poetry run python tests/test_deepseek_ocr_provider_simple.py
```

### Run All Provider Tests

```bash
poetry run python test_provider_outputs.py
```

### Check SDK Versions

```bash
poetry run python scripts/check_provider_sdks.py
```

## What to Look For

### ✅ Success Indicators

1. **Content Extraction**: Should see actual text like:
   ```
   📄 FIRST PAGE:
      Content length: 1399 chars
      First 500 chars:
   
   PROCESS OVERVIEW
   Nuance Voice ID: Quick Reference Guide
   Answering Calls Using the Voice ID System...
   ```

2. **Provider-Specific Data**: Should contain actual content:
   ```
   📦 PROVIDER_SPECIFIC:
      ✅ Has 'content' field: 1399 chars
   ```

### ❌ Failure Indicators

1. **Reference IDs Instead of Content**:
   ```json
   {
     "file_id": "file_xyz",
     "parse_id": "parse_abc"
   }
   ```

2. **Empty or Missing Content**:
   ```
   ❌ No actual content found in provider_specific!
   ```

3. **SDK/Import Errors**:
   ```
   ❌ tensorlake package not installed
   ```

## Troubleshooting

### TensorLake Issues

**Problem**: SSL certificate errors
```bash
SSLError: certificate verify failed
```

**Solution**: The provider now handles this automatically by using `certifi` bundle.

**Problem**: No content extracted (showing parse_id instead)
```
Document processed (parse_id: parse_xxx)
```

**Solution**: The provider now explicitly calls `get_parsed_result()` after waiting for completion to ensure all chunks are fetched.

### Gemini Issues

**Problem**: API key not found
```bash
GOOGLE_API_KEY or GEMINI_API_KEY not set
```

**Solution**: Set either `GOOGLE_API_KEY` or `GEMINI_API_KEY` in `.env`

### PaddleOCR Issues

**Problem**: Package not installed
```bash
paddleocr not installed
```

**Solution**: 
```bash
poetry add paddleocr
```

**Note**: PaddleOCR runs locally and may take longer on first run as it downloads models.

### DeepSeek-OCR Issues

**Problem**: API key not set
```bash
DEEPSEEK_API_KEY not set
```

**Solution**: Get API key from DeepSeek and add to `.env`

## Integration with E2E Tests

After unit tests pass, run full end-to-end tests:

```bash
# Test specific provider with full workflow
poetry run pytest tests/test_document_processing_v2.py::test_document_upload_v2_with_tensorlake_provider -v -s

# Or run all provider tests
poetry run python tests/run_v1_tests.py
```

## Expected Output

All tests should show:

```
================================================================================
SUMMARY
================================================================================
   TensorLake: ✅ PASSED
   Gemini: ✅ PASSED
   PaddleOCR: ✅ PASSED
   DeepSeek-OCR: ✅ PASSED
================================================================================
```

## Notes

- **TensorLake**: Uses official SDK, extracts from `result.chunks[].content`
- **Gemini**: Uses Google's generativeai SDK, extracts from response text
- **PaddleOCR**: Runs locally, no API calls, extracts from OCR results
- **DeepSeek-OCR**: HTTP API, extracts from response JSON

All providers must return actual text content, not just reference IDs!

