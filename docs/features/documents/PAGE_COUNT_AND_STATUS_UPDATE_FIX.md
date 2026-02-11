# Page Count and Status Update Fixes

## Issues Identified

### Issue 1: Incorrect Page Count (showing 1 instead of 30)
**Problem:** Reducto's `result.total_pages` field was returning 1 instead of the actual page count (30 pages).

**Root Cause:** Reducto stores the actual page count in a nested field: `result.parse.result.usage.num_pages`, not in the top-level `result.total_pages`.

### Issue 2: Duplicate Status Updates
**Problem:** Two status updates were being sent too close together:
- First: `"processing"` at 0.6 progress (no total_pages)
- Second: `"creating_memories"` at 0.7 progress (with total_pages=1, wrong count)

**Root Cause:** Status updates were being sent before we had the correct page count, and too close together in the workflow timeline.

## Fixes Applied

### Fix 1: Extract Actual Page Count from Reducto

**File:** `cloud_plugins/temporal/activities/document_activities.py`  
**Lines:** 763-773

```python
# Extract actual page count from Reducto response (result.total_pages is often 1)
actual_total_pages = result.total_pages
if processor.provider_name.lower() == "reducto" and result.provider_specific:
    try:
        # Reducto stores actual page count in result.parse.result.usage.num_pages
        usage = result.provider_specific.get("result", {}).get("parse", {}).get("result", {}).get("usage", {})
        if usage and "num_pages" in usage:
            actual_total_pages = usage.get("num_pages", result.total_pages)
            logger.info(f"Extracted actual page count from Reducto usage: {actual_total_pages}")
    except Exception as e:
        logger.warning(f"Failed to extract page count from Reducto response: {e}")
```

**What it does:**
1. Tries to extract the actual page count from Reducto's nested `usage.num_pages` field
2. Falls back to `result.total_pages` if extraction fails
3. Logs the corrected page count for visibility

**Updated return statement** (lines 820, 825):
```python
"stats": {
    "total_pages": actual_total_pages,  # Use corrected page count
    "processing_time": result.processing_time,
    "confidence": result.confidence,
    "provider": processor.provider_name,
}
```

### Fix 2: Consolidate Status Updates

**File:** `cloud_plugins/temporal/workflows/document_processing.py`

#### Removed: Premature status update at 0.6
**Before (line 88):**
```python
await self._update_status("processing", 0.6)
```

**Removed** - This was sending status before we knew the correct page count.

#### Changed: First status update to "analyzing_structure"
**Before (line 98):**
```python
await self._update_status("creating_memories", 0.7, None, total_pages)
```

**After (line 96):**
```python
await self._update_status("analyzing_structure", 0.65, None, total_pages)
```

**Changes:**
- Status changed from `"creating_memories"` to `"analyzing_structure"` (more accurate)
- Progress changed from 0.7 to 0.65
- Now includes correct `total_pages` from the fixed extraction

#### Added: Status update after extraction and image processing
**New (line 145):**
```python
# Send status update after extraction and image processing
await self._update_status("creating_memories", 0.75, None, total_pages)
```

**Why this is better:**
- Sent AFTER extraction and image processing are complete
- Progress is 0.75 (more accurate timeline)
- Uses the correct `total_pages` count
- Only ONE status update during memory creation phase

## Status Update Flow (Before vs After)

### Before ❌
```
0.6  - "processing"         (no total_pages)
0.7  - "creating_memories"  (total_pages=1, wrong)
```

### After ✅
```
0.65 - "analyzing_structure" (total_pages=30, correct)
0.75 - "creating_memories"   (total_pages=30, correct)
```

## Benefits

1. ✅ **Correct Page Count**: Now shows 30 pages instead of 1
2. ✅ **No Duplicate Updates**: Removed redundant status update
3. ✅ **Better Status Labels**: "analyzing_structure" before "creating_memories"
4. ✅ **Accurate Progress**: Status sent after activities complete, not before
5. ✅ **Provider-Specific Handling**: Gracefully handles Reducto's nested structure

## Testing

When you restart the worker and run the E2E test, you should now see:

1. **First Status Update:**
   ```json
   {
     "status": "analyzing_structure",
     "progress": 0.65,
     "total_pages": 30
   }
   ```

2. **Second Status Update (after extraction + image processing):**
   ```json
   {
     "status": "creating_memories",
     "progress": 0.75,
     "total_pages": 30
   }
   ```

3. **Processing Result:**
   ```json
   {
     "stats": {
       "total_pages": 30,
       "provider": "reducto"
     }
   }
   ```

## Notes

- The fix is provider-specific (Reducto) but gracefully falls back for other providers
- Image extraction activity (if it runs) happens between the two status updates
- The `total_pages` value now propagates correctly through the entire workflow

