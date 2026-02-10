# Provider Test Results

## Summary

Provider tests have been created and tested. Here's the current status:

| Provider | Test Status | Requirements | Notes |
|----------|-------------|--------------|-------|
| TensorLake | ✅ **PASSING** | `TENSORLAKE_API_KEY` | SDK installed, extracts 1399 chars successfully |
| Gemini Vision | ⚠️ **NEEDS SETUP** | `GOOGLE_API_KEY` or `GEMINI_API_KEY`, `pymupdf` | Crashes on initialization - needs dependency check |
| PaddleOCR | ⚠️ **NEEDS INSTALL** | `paddleocr` package | Not installed - runs locally, no API key needed |
| DeepSeek-OCR | ⚠️ **NEEDS API KEY** | `DEEPSEEK_API_KEY` | Code ready, just needs API key |

## Test Results Details

### ✅ TensorLake Provider
**Status**: PASSING  
**Command**: `poetry run python tests/test_tensorlake_sdk_simple.py`

```
✅ Successfully extracted 1399 chars from document
📄 FIRST PAGE:
   Content length: 1399 chars
   First 500 chars:

PROCESS OVERVIEW
Nuance Voice ID: Quick Reference Guide
Answering Calls Using the Voice ID System...
```

**What Works**:
- SDK initialization
- File upload
- Document parsing
- Content extraction from chunks
- Actual text content (not just IDs)

---

### ⚠️ Gemini Vision Provider
**Status**: NEEDS SETUP  
**Command**: `poetry run python tests/test_gemini_provider_simple.py`

**Issue**: Crashes on initialization (exit code 139)

**Requirements**:
1. Install SDK:
   ```bash
   poetry add google-generativeai
   ```

2. Install PyMuPDF for PDF processing:
   ```bash
   poetry add pymupdf
   ```

3. Set API key:
   ```bash
   # Add to .env
   GOOGLE_API_KEY=your_api_key
   # OR
   GEMINI_API_KEY=your_api_key
   ```

**Next Steps**:
- Check if packages are installed: `poetry show google-generativeai pymupdf`
- Verify API key is set in `.env`
- Re-run test after setup

---

### ⚠️ PaddleOCR Provider
**Status**: NEEDS INSTALL  
**Command**: `poetry run python tests/test_paddleocr_provider_simple.py`

**Result**:
```
❌ PaddleOCR not installed. Install with: pip install paddleocr
```

**Requirements**:
1. Install PaddleOCR:
   ```bash
   poetry add paddleocr
   ```

**Notes**:
- Runs locally (no API key needed)
- Downloads models on first run
- May be slower initially
- Good for offline OCR

---

### ⚠️ DeepSeek-OCR Provider  
**Status**: NEEDS API KEY  
**Command**: `poetry run python tests/test_deepseek_ocr_provider_simple.py`

**Result**:
```
❌ DEEPSEEK_API_KEY not set
```

**Requirements**:
1. Get API key from DeepSeek
2. Add to `.env`:
   ```bash
   DEEPSEEK_API_KEY=your_api_key
   ```

**Notes**:
- Uses HTTP API (no special SDK needed)
- `httpx` already installed
- Ready to test once API key is set

---

## Installation Commands

### Install All Provider SDKs

```bash
# TensorLake (already installed)
poetry update tensorlake

# Gemini
poetry add google-generativeai pymupdf

# PaddleOCR  
poetry add paddleocr

# DeepSeek (no SDK needed, httpx already installed)
```

### Check Installed Versions

```bash
poetry run python scripts/check_provider_sdks.py
```

---

## Environment Variables Required

Add to your `.env` file:

```bash
# TensorLake (working)
TENSORLAKE_API_KEY=your_tensorlake_api_key

# Gemini (needs setup)
GOOGLE_API_KEY=your_google_api_key
# OR
GEMINI_API_KEY=your_gemini_api_key

# DeepSeek (needs setup)
DEEPSEEK_API_KEY=your_deepseek_api_key

# PaddleOCR (no key needed - runs locally)
```

---

## Running All Tests

### Individual Tests

```bash
# TensorLake
poetry run python tests/test_tensorlake_sdk_simple.py

# Gemini (after setup)
poetry run python tests/test_gemini_provider_simple.py

# PaddleOCR (after install)
poetry run python tests/test_paddleocr_provider_simple.py

# DeepSeek (after API key)
poetry run python tests/test_deepseek_ocr_provider_simple.py
```

### All Tests (batch runner)

```bash
poetry run python tests/run_provider_tests.py
```

---

## Next Steps

1. **For Gemini**:
   - Install dependencies: `poetry add google-generativeai pymupdf`
   - Set API key in `.env`
   - Re-run test

2. **For PaddleOCR**:
   - Install: `poetry add paddleocr`
   - Run test (no API key needed)

3. **For DeepSeek**:
   - Get API key from DeepSeek
   - Add to `.env`
   - Run test

4. **After All Pass**:
   - Run end-to-end tests for each provider
   - Verify actual content extraction in workflows
   - Test with real documents

---

## Expected Output When All Pass

```
================================================================================
FINAL SUMMARY
================================================================================
  TensorLake          ✅ PASSED
  Gemini              ✅ PASSED
  PaddleOCR           ✅ PASSED
  DeepSeek-OCR        ✅ PASSED
================================================================================
✅ All provider tests passed!
```

---

## Troubleshooting

### Gemini Crashes (Exit 139)
- Install `pymupdf`: `poetry add pymupdf`
- Check API key is valid
- Try with a smaller test file first

### PaddleOCR Slow
- First run downloads models (~100MB)
- Subsequent runs are faster
- Consider increasing timeout for first run

### DeepSeek API Errors
- Verify API key is correct
- Check API quota/limits
- Ensure network access is available

### TensorLake SSL Issues
- Already fixed in code (uses `certifi` bundle)
- If issues persist, check `SSL_CERT_FILE` in `.env`

