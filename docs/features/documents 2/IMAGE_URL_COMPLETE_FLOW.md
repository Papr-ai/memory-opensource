# Image URL Preservation - Complete Flow

## Summary
Images are now properly handled throughout the entire document processing pipeline, from provider extraction through LLM enhancement to final memory storage.

## Flow

### 1. Provider Extraction (`provider_adapter.py`)
**Lines 150-195**

When Reducto (or other providers) returns image/figure blocks:

```python
# Extract image URL from provider response
image_url = block.url or block.image_url  # May be None

# Create markdown image content
if image_url:
    image_content = f"![{description}]({image_url})"
else:
    image_content = f"![{description}](#)"  # Placeholder

# Add description as formatted text
image_content = f"{image_content}\n\n*{description}*"

# Create ImageElement
ImageElement(
    content=image_content,  # Markdown format
    image_url=image_url,    # Stored separately (may be None)
    image_description=description
)
```

**Result:**
- Content includes markdown image syntax (visible/clickable)
- URL stored in element attribute for programmatic access
- Works even when URL is `null` (uses `#` placeholder)

### 2. Content Consolidation (`llm_memory_generator.py:_consolidate_small_elements`)
**Lines 720-780**

Images are **kept separate** from text consolidation:

```python
# ImageElements are never consolidated - always processed individually
if element.content_type == ContentType.IMAGE:
    result.append(element)  # Keep as-is
```

**Result:**
- Each image gets its own LLM call for rich metadata generation
- Image markdown and URLs are preserved

### 3. LLM Enhancement (`llm_memory_generator.py:_parse_llm_response`)
**Lines 468-476**

Original image attributes are explicitly preserved:

```python
# ALWAYS preserve ImageElement attributes (even if None)
if hasattr(content_element, 'image_url'):
    enhanced_metadata.customMetadata['image_url'] = content_element.image_url

if hasattr(content_element, 'image_description'):
    enhanced_metadata.customMetadata['image_description'] = content_element.image_description

if hasattr(content_element, 'image_hash'):
    enhanced_metadata.customMetadata['image_hash'] = content_element.image_hash

if hasattr(content_element, 'ocr_text'):
    enhanced_metadata.customMetadata['ocr_text'] = content_element.ocr_text
```

**Result:**
- Image URL appears in `metadata.customMetadata.image_url` (even if `null`)
- Image description preserved
- Original markdown content from step 1 is retained

### 4. Final Memory Request
**Output: `AddMemoryRequest`**

```json
{
  "content": "![Google Logo](#)\n\n*Google*",
  "metadata": {
    "customMetadata": {
      "image_url": null,
      "image_description": "Google",
      "image_hash": null,
      "ocr_text": null,
      "bbox": {
        "page": 1,
        "left": 0.103,
        "top": 0.050,
        "width": 0.095,
        "height": 0.022
      }
    }
  }
}
```

## Key Features

### ✅ Markdown in Content
- Images always appear as markdown: `![description](url)` or `![description](#)`
- Visible and clickable in markdown viewers
- Included in embeddings for semantic search

### ✅ URL in Metadata
- Always present in `customMetadata.image_url` (even if `null`)
- Programmatic access for downloading/processing
- Preserved through all pipeline stages

### ✅ Bbox Coordinates
- Bounding box coordinates from provider
- Useful for cropping original PDF
- Page number for multi-page documents

### ✅ Null Handling
- Works gracefully when provider doesn't return URLs
- Uses `#` placeholder in markdown
- Still preserves description and bbox

## Testing

The real Reducto file (`b1ee8b3479b29f40964bdaa830163b19_provider_result_f8141f7d-88ba-4145-925c-0b025b22d6c7.json`) has:
- **All image URLs are `null`** (Reducto didn't extract them for this PDF)
- 10 figure elements with descriptions
- Bbox coordinates available

**Expected behavior:**
- Content: `![Google](#)\n\n*Google*`
- Metadata: `image_url: null`, `image_description: "Google"`, `bbox: {...}`

## Why This Matters

1. **Embeddings:** Markdown content is embedded, so searches for "Google logo" will find this memory
2. **Display:** When showing memories to users, markdown renders as images (if URL available)
3. **Extraction:** If URL is available later, it's in metadata for download/processing
4. **Context:** Bounding boxes allow re-extracting images from original PDFs if needed

## Future Enhancements

If we want to extract images when provider doesn't provide URLs:
1. Use bbox coordinates to crop from original PDF
2. Upload cropped image to storage (Parse Server, S3, etc.)
3. Update `image_url` in metadata with uploaded URL
4. Re-embed memory with updated content

This could be a separate workflow activity that runs after initial processing.
