# Hierarchical Chunking & LLM Memory Generation Flow

## Overview
This document explains the complete document processing pipeline from Reducto extraction to final memory creation.

---

## Stage 1: Provider Extraction (Reducto)
**Location**: Reducto API → Parse Server

**Input**: 21-page PDF (QPNC83-106 Instruction Manual.pdf)

**Output**: 247 structured elements
- ~194 text blocks (paragraphs, headings, lists)
- 14 tables (with structured_data HTML)
- 39 images (with image_url and descriptions)

**Storage**: Stored in Parse Post's `provider_extraction` field (compressed JSON)

---

## Stage 2: Hierarchical Chunking
**Location**: `document_activities.py` → `chunk_document_elements()` (lines 1651-1910)
**Temporal Activity**: `chunk_document_elements`

### What It Does:
1. **Context Enrichment** (`hierarchical_chunker.py` lines 489-583)
   - For each table/image, extract 400 chars before + 400 chars after
   - Based on Anthropic 2025 research: context improves retrieval by 67%
   - Result: Tables and images now contain surrounding text for better semantic understanding

2. **Semantic Grouping** (`hierarchical_chunker.py` lines 718-807)
   - Groups text elements by section while maintaining document structure
   - Preserves tables and images as separate chunks (they already have context!)
   - Uses `max_chunk_size=6000` (target: 1-2 pages per chunk)
   - Uses `min_chunk_size=1000` (avoid tiny fragments)

### Input: 247 elements
- 194 text blocks
- 14 tables
- 39 images

### Output: 96 chunks ✅
- **43 text chunks** (194 text blocks → 43 semantically grouped chunks, 77% reduction)
- **14 table chunks** (preserved separately, each with 400 chars context)
- **39 image chunks** (preserved separately, each with 400 chars context)

**Example of context enrichment:**
```
BEFORE:
Table: <table><tr><th>Reference</th><th>Name</th></tr>...</table>

AFTER:
[Context before: BEFORE CARRYING OUT THE OPERATION TEST, READ CAREFULLY AND ACQUIRE A GOOD KNOWLEDGE OF THE COMMAND FUNCTIONS...]

<table><tr><th>Reference</th><th>Name</th></tr>...</table>

[Context after: The control panel features a multi-segment LED status matrix...]
```

**Storage**: Stored back to Parse Post's `chunked_extraction` field (replaces provider_extraction)

**Logs to expect:**
```
Hierarchical chunking: 247 elements → 96 chunks (61.1% reduction)
Chunked element types: {'text': 43, 'table': 14, 'image': 39}
```

---

## Stage 3: LLM Memory Generation
**Location**: `document_activities.py` → `generate_llm_optimized_memory_structures()` (lines 2209-2513)
**Temporal Activity**: `generate_llm_optimized_memory_structures`

### Step 3.1: Consolidation (`llm_memory_generator.py` lines 835-895)

**THE FIX WE JUST IMPLEMENTED:**
- **OLD BEHAVIOR**: Process elements in document order → tables/images break consolidation batches
  ```
  [text1, text2, TABLE, text3, text4, IMAGE, text5, ...]
         ↑            ↑                     ↑
     Merge these  BREAK!  Merge these  BREAK!
  ```
  Result: 43 text chunks stay as 43 memories (0% reduction)

- **NEW BEHAVIOR**: Group by type FIRST, then consolidate all text together
  ```
  Group: [text1, text2, text3, text4, text5, ...43 text chunks] + [14 tables] + [39 images]
               ↑ Consolidate these into ~5-10 big memories    ↑           ↑
                                                          Keep separate  Keep separate
  ```
  Result: 43 text chunks → ~5-10 consolidated memories (77-88% reduction!)

**What `_consolidate_small_elements()` does now:**

1. **Separate by type:**
   ```python
   text_elements = [e for e in content_elements if e.content_type.value == "text"]    # 43 chunks
   table_elements = [e for e in content_elements if e.content_type.value == "table"]   # 14 chunks
   image_elements = [e for e in content_elements if e.content_type.value == "image"]   # 39 chunks
   ```

2. **Consolidate text elements:**
   - Merge consecutive text chunks up to 6000 chars each
   - Keep large chunks (>4500 chars) separate
   - Result: 43 text chunks → **~5-10 consolidated text memories**

3. **Preserve tables/images:**
   - Tables already have context (400 chars before/after) ✅
   - Images already have context (400 chars before/after) ✅
   - Keep them as separate memories for targeted retrieval

4. **Combine:**
   ```python
   consolidated = consolidated_text + table_elements + image_elements + chart_elements
   ```

### Step 3.2: LLM Enhancement (`llm_memory_generator.py` lines 912-953)

For each consolidated memory:
1. **Generate metadata** (title, topics, entities, keywords)
2. **Create query patterns** (example questions users might ask)
3. **Identify relationships** (links to other content)
4. **Add document context** (document title, type, page range)

**Models used (in order of preference):**
1. **Groq** (llama-3.3-70b-versatile with JSON schema) - FREE, fast
2. **Gemini** (1.5 Flash) - Fallback
3. **OpenAI** (GPT-4o-mini) - Last resort

---

## Final Result: Memory Requests

### Expected Output for 21-page manual:

**~58-63 total memories** (instead of 96!)

Breakdown:
- **~5-10 text memories** (consolidated from 43 text chunks)
  - Each contains ~6000 chars (~1-2 pages of text)
  - LLM-enhanced with rich metadata
  
- **14 table memories** (preserved from 14 table chunks)
  - Each contains: table HTML + 400 chars context before/after
  - LLM identifies table structure, key data points, relationships
  
- **39 image memories** (preserved from 39 image chunks)
  - Each contains: image URL + description + 400 chars context
  - LLM generates visual descriptions, identifies objects, creates searchable metadata

### Example Memory Structure:

**Text Memory (consolidated):**
```json
{
  "content": "Combined text from multiple sections...",
  "title": "Control Panel Operation Instructions - Pages 8-10",
  "metadata": {
    "topics": ["control panel", "operation", "instructions"],
    "entities": [{"name": "Digital Controller", "type": "component"}],
    "query_patterns": [
      "How do I operate the control panel?",
      "What buttons are on the control panel?"
    ],
    "page_range": "8-10",
    "consolidated_from": ["text_001", "text_002", "text_003"]
  }
}
```

**Table Memory (with context):**
```json
{
  "content": "[Context before: BEFORE CARRYING OUT THE OPERATION TEST...]\n\n<table><tr><th>Reference</th><th>Name</th></tr>...</table>\n\n[Context after: The control panel features...]",
  "title": "Control Panel Button Reference Table - Page 9",
  "metadata": {
    "content_type": "table",
    "has_context_enrichment": true,
    "context_before_length": 400,
    "context_after_length": 400,
    "topics": ["control panel", "button reference", "user interface"],
    "query_patterns": [
      "What does button 7 do?",
      "Show me the control panel button layout"
    ],
    "page_number": 9
  }
}
```

**Image Memory (with context):**
```json
{
  "content": "[Context before: Fig. 9a shows the command and control panel...]\n\n![Control Panel Diagram](https://parse.../image.png)\n\n*Control panel with digital display and four buttons*\n\n[Context after: Press the SET button to configure...]",
  "title": "Control Panel Diagram - FIG. 9a",
  "metadata": {
    "content_type": "image",
    "has_context_enrichment": true,
    "image_url": "https://parse.../image.png",
    "topics": ["control panel", "diagram", "visual reference"],
    "query_patterns": [
      "Show me what the control panel looks like",
      "Where is the SET button located?"
    ],
    "page_number": 9
  }
}
```

---

## Performance Improvements

### Before Fix:
- Stage 1: 247 elements (Reducto)
- Stage 2: 96 chunks (61% reduction) ✅
- Stage 3: **96 memories** (0% reduction) ❌
- **Total reduction: 61%**

### After Fix:
- Stage 1: 247 elements (Reducto)
- Stage 2: 96 chunks (61% reduction) ✅
- Stage 3: **~58-63 memories** (40% reduction) ✅
- **Total reduction: 75-76%** 🎉

---

## Why This Approach Works

### 1. **Context-Enriched Tables/Images**
- Research-backed: Anthropic 2025 showed 67% better retrieval with context
- Users can search "table showing button functions" and get the table WITH surrounding explanation
- Better than standalone tables that lack semantic context

### 2. **Consolidated Text Memories**
- Users rarely search for tiny fragments
- 1-2 page chunks provide complete semantic units (introduction → explanation → conclusion)
- Reduces embedding storage costs and improves search quality

### 3. **Semantic Grouping**
- Document structure is preserved (sections stay together)
- Related content is co-located for better retrieval
- LLM can generate better metadata with more context

### 4. **Optimal Memory Count**
- 21 pages → ~58-63 memories ≈ **3 memories per page**
- Balance between granularity (findability) and consolidation (coherence)
- Matches research recommendations for document chunking

---

## Logs to Monitor

### Stage 2 - Hierarchical Chunking:
```
Starting hierarchical chunking for 247 elements
Enriched 53/247 elements with surrounding context
Hierarchical chunking: 247 elements → 96 chunks (grouped by section)
Hierarchical chunking complete: 96 chunks from 247 elements (61.1% reduction)
Chunked element types: {'text': 43, 'table': 14, 'image': 39}
```

### Stage 3 - LLM Consolidation (NEW LOGS):
```
Generating LLM-optimized memory structures for 96 elements
Grouping elements by type: 43 text, 14 tables, 39 images, 0 charts
After consolidation: 58 elements (original: 96)
  - Text: 43 → 5 memories (38 merged)
  - Tables: 14 memories (preserved with context)
  - Images: 39 memories (preserved with context)
  - Charts: 0 memories (preserved with context)
Consolidated 96 elements into 58 memories (reduction: 39.6%)
```

---

## Testing the Fix

Run a new document upload and check the logs for:

1. ✅ **Stage 2**: `96 chunks` (with breakdown of text/table/image)
2. ✅ **Stage 3**: `~58-63 memories` (with consolidation stats)
3. ✅ **Final**: Parse Post should have `~58-63` linked Memory records

**Expected memory count formula:**
```
Final Memories = (Text chunks / 6-8) + Tables + Images
               = (43 / 6-8) + 14 + 39
               = 5-7 + 14 + 39
               = 58-60 memories
```

For your 21-page instruction manual, you should see **~58-63 total memories** instead of 96! 🚀

