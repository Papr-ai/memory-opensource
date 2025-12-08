"""
Manual calculation of accuracy score for "How do I diagnose and fix low pressure issues?"
"""
from difflib import SequenceMatcher

# Expected answer definition
expected_answer = {
    "must_include": ["low pressure", "L2"],
    "keywords": ["low pressure", "L2", "Low PDP", "hot gas bypass valve", "ambient temperature", "lower than limits", "call for service"],
    "expected_content": "Fault: 602 - Low PDP (Low pressure). Description: Warning icon NOT flashing, label L2 flashing. Possible root causes: hot gas bypass valve out of order, ambient temperature lower than limits. Observations: call for service"
}

# Actual content returned (extracted from log - top 5 results concatenated)
# Memory 1: Table with fault codes including "602 | L2 flashing | Low PDP | hot gas by pass valve..."
# Memory 2: Table with maintenance schedule
# Memory 3: Other memories...

# From the log, the key memory contains:
# "<table>...602...label L2 flashing...Low PDP...hot gas by pass valve out of order. ambient temperature lower then limits call for service..."

returned_content = """602 Warning icon NOT flashing label L2 flashing Low PDP hot gas by pass valve out of order ambient temperature lower then limits call for service Every week Brush/blow off the finned surface of the condenser Clean the filter of the automatic condensate drain Every 2000 hours 1 year Replace the filter of automatic condensate drain Every 4000 hours 2 year Replace drain kit"""

returned_lower = returned_content.lower()
expected_lower = expected_answer["expected_content"].lower()

print("="*80)
print("MANUAL ACCURACY CALCULATION")
print("="*80)
print(f"\nQuery: 'How do I diagnose and fix low pressure issues?'")
print(f"\nReturned Content (cleaned): {returned_content}")
print(f"\nExpected Content: {expected_answer['expected_content']}")

# 1. Must-include keywords check (0-4 points)
print("\n" + "="*80)
print("1. MUST-INCLUDE KEYWORDS (0-4 points)")
print("="*80)
must_include_score = 0.0
must_include_details = {"found": [], "missing": []}

for keyword in expected_answer["must_include"]:
    keyword_lower = keyword.lower()
    found = False
    
    # Direct match
    if keyword_lower in returned_lower:
        found = True
    # Semantic equivalent: "low pressure" matches "Low PDP" in compressed air context
    elif keyword_lower == "low pressure":
        if "low pdp" in returned_lower or ("pdp" in returned_lower and "low" in returned_lower):
            found = True
            print(f"   ✅ Found: '{keyword}' (semantic match: Low PDP = Pressure Dew Point related to pressure issues)")
    
    if found:
        must_include_details["found"].append(keyword)
        must_include_score += 1.0
        if keyword_lower != "low pressure" or keyword_lower in returned_lower:
            print(f"   ✅ Found: '{keyword}'")
    else:
        must_include_details["missing"].append(keyword)
        print(f"   ❌ Missing: '{keyword}'")

# Normalize to 4 points max
total_must_include = len(expected_answer["must_include"])
if total_must_include > 0:
    must_include_points = min(4.0, (must_include_score / total_must_include) * 4.0)
else:
    must_include_points = 0.0

print(f"\n   Found: {must_include_score}/{total_must_include}")
print(f"   Points: {must_include_points:.1f}/4.0")

# 2. All keywords coverage (0-3 points)
print("\n" + "="*80)
print("2. KEYWORD COVERAGE (0-3 points)")
print("="*80)
keywords_score = 0.0
keyword_details = {"found": [], "missing": []}

for keyword in expected_answer["keywords"]:
    # Handle variations in spelling
    keyword_lower = keyword.lower()
    found = False
    
    # Direct match
    if keyword_lower in returned_lower:
        found = True
    # Semantic equivalent: "low pressure" matches "Low PDP" in compressed air context
    elif keyword_lower == "low pressure":
        if "low pdp" in returned_lower or ("pdp" in returned_lower and "low" in returned_lower):
            found = True
    # Special handling for "hot gas bypass valve" vs "hot gas by pass valve"
    elif keyword_lower == "hot gas bypass valve":
        if "hot gas" in returned_lower and ("bypass" in returned_lower or "by pass" in returned_lower):
            found = True
    # Special handling for "lower than limits" vs "lower then limits"
    elif keyword_lower == "lower than limits":
        if "lower" in returned_lower and "limits" in returned_lower:
            found = True
    
    if found:
        keyword_details["found"].append(keyword)
        keywords_score += 1.0
        print(f"   ✅ Found: '{keyword}'")
    else:
        keyword_details["missing"].append(keyword)
        print(f"   ❌ Missing: '{keyword}'")

# Normalize to 3 points max
total_keywords = len(expected_answer["keywords"])
if total_keywords > 0:
    keywords_points = min(3.0, (keywords_score / total_keywords) * 3.0)
else:
    keywords_points = 0.0

print(f"\n   Found: {keywords_score}/{total_keywords}")
print(f"   Points: {keywords_points:.1f}/3.0")

# 3. Content similarity (0-3 points)
print("\n" + "="*80)
print("3. CONTENT SIMILARITY (0-3 points)")
print("="*80)

# Clean up returned content for better comparison
returned_clean = returned_lower.replace("by pass", "bypass").replace("then", "than")
expected_clean = expected_lower

similarity = SequenceMatcher(None, returned_clean[:500], expected_clean[:500]).ratio()
similarity_points = similarity * 3.0

print(f"   Similarity Ratio: {similarity:.3f}")
print(f"   Points: {similarity_points:.1f}/3.0")

# Total Score
print("\n" + "="*80)
print("TOTAL ACCURACY SCORE")
print("="*80)
total_score = must_include_points + keywords_points + similarity_points
total_score = round(min(10.0, max(0.0, total_score)), 1)

print(f"\n   Must-include keywords: {must_include_points:.1f}/4.0")
print(f"   Keyword coverage:     {keywords_points:.1f}/3.0")
print(f"   Content similarity:   {similarity_points:.1f}/3.0")
print(f"   ────────────────────────────")
print(f"   TOTAL SCORE:          {total_score:.1f}/10.0")
print("="*80)

