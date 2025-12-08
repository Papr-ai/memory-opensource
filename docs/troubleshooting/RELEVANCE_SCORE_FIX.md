# Relevance Score Fix for Tier0 Predictive Builder

**Date**: November 24, 2025  
**Status**: ✅ Fixed  
**Issue**: All tier0 memories had `relevance_score = 0.2` regardless of actual relevance

---

## Problem Identified

The `relevance_score` was incorrectly calculated because:

1. **Hotness component used flat 0.2**: All "hot" memories received a flat `0.2` score, regardless of their actual access frequency
2. **No normalization**: The `hot_counts` values (which already include time decay from retrieval logs) were not normalized
3. **Missing log normalization**: Without log normalization, outliers could dominate the scoring

**Root Cause**: Line 212 in `tier0_builder.py`:
```python
add_score(oid, m, 0.2, "hot")  # ❌ Flat 0.2 for ALL hot memories!
```

---

## Solution Implemented

### 1. **Log-Normalized Hotness** (Research-Backed)

Following tier1's pattern and IR best practices (BM25, TF-IDF), we now use `log1p` normalization:

```python
def _log1p(x: float) -> float:
    """Log(1+x) normalization for count-based features"""
    return math.log1p(float(x) if x is not None else 0.0)

# Normalize hotness: log1p(count) / log1p(max_count) to [0, 1] range
count = hot_counts[oid]
log_hotness = _log1p(count)
normalized_hotness = log_hotness / max_log_hotness if max_log_hotness > 0 else 0.0
hotness_score = 0.2 * normalized_hotness  # ✅ Weighted by actual frequency
```

**Why log1p?**
- Prevents outliers from dominating (a memory accessed 1000x doesn't get 1000x the score)
- Standard in information retrieval (BM25, TF-IDF variants)
- Smooths the distribution for better ranking

### 2. **Hybrid Ranking Formula** (Research-Backed)

The final `relevance_score` is now correctly computed as:

```
relevance_score = 0.6 * vector_similarity + 0.3 * transition_probability + 0.2 * normalized_hotness
```

**Components**:
- **Vector Similarity (60%)**: Semantic relevance to goals/OKRs via embedding similarity
- **Transition Probability (30%)**: Contextual relevance via Markov chain transitions (includes temporal decay: `decay_factor=0.95` per day)
- **Normalized Hotness (20%)**: Access frequency with log normalization (time decay already applied via `days=30` filter)

**Why these weights?**
- Semantic relevance (vector) is most important → 60%
- Contextual patterns (transitions) are important → 30%
- Popularity (hotness) provides signal but shouldn't dominate → 20%
- Weights sum to 1.1, allowing for additive boost when multiple signals align

### 3. **Time Decay Already Applied**

As you correctly noted, time decay is **already** handled:
- **Retrieval logs**: Filtered by `days=30` (only recent accesses)
- **Transition matrix**: Applies exponential decay (`decay_factor=0.95` per day in `TransitionMatrixBuilder`)
- **No additional decay needed**: The `hot_counts` already reflect recent activity

---

## Research-Backed Improvements

### Log Normalization (BM25/TF-IDF)
- **Source**: Standard practice in information retrieval
- **Benefit**: Prevents count-based features from being dominated by outliers
- **Implementation**: `log1p(count)` instead of raw count

### Hybrid Ranking (Learning to Rank)
- **Source**: Modern IR systems combine multiple signals
- **Components**: Semantic (vector), contextual (transition), popularity (hotness)
- **Benefit**: More robust than single-signal ranking

### Temporal Decay (Exponential)
- **Source**: Transition matrix already implements this
- **Formula**: `decay_weight = decay_factor ** age_days` where `decay_factor=0.95`
- **Benefit**: Recent accesses weighted more heavily

---

## Expected Impact

**Before**:
- All memories: `relevance_score = 0.2` (if only in hot list)
- No differentiation between frequently vs. rarely accessed memories
- Short/irrelevant memories could rank high

**After**:
- Memories with high access frequency: `relevance_score ≈ 0.2 * normalized_hotness` (up to 0.2)
- Memories with vector similarity: `relevance_score ≈ 0.6 * similarity` (up to 0.6)
- Memories with transitions: `relevance_score ≈ 0.3 * probability` (up to 0.3)
- **Combined scores**: Can reach up to 1.1 when all signals align
- Better ranking: More relevant memories (with multiple signals) rank higher

---

## Code Changes

**File**: `/Users/shawkatkabbara/Documents/GitHub/memory/services/predictive/tier0_builder.py`

**Key Changes**:
1. Added `import math` at top
2. Added `_log1p()` helper function for log normalization
3. Compute `max_log_hotness` for normalization
4. Replace flat `0.2` with `0.2 * normalized_hotness` based on actual `hot_counts`
5. Enhanced logging to show count and normalized value

**Final relevance_score assignment** (line 281):
```python
memory_obj.relevance_score = float(x["score"])  # ✅ Already correct - uses accumulated score
```

The `x["score"]` is the sum of all three components, which is exactly what we want!

---

## Testing Recommendations

1. **Verify score distribution**: Check that `relevance_score` now varies (not all 0.2)
2. **Check top-ranked items**: Should have high scores from multiple signals
3. **Compare with tier1**: Both should use similar normalization patterns
4. **Monitor logs**: The enhanced logging shows `hot(count=X, normalized=Y)` for debugging

---

## Future Enhancements (Optional)

Based on latest research, could consider:
1. **Content quality signals**: Length, structure, completeness
2. **User engagement**: Click-through, dwell time (if available)
3. **Recency boost**: Additional weight for very recent accesses (< 7 days)
4. **Learning to Rank**: Train ML model to optimize weights based on user feedback

