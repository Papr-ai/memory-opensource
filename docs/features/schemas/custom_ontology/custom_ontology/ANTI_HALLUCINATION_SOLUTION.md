# Anti-Hallucination Solution for LLM Structured Outputs

## Problem

When using OpenAI's structured output feature (gpt-4o-mini) to extract nodes/relationships from memory content, the API requires all properties in the schema to be marked as `required`. This forces the LLM to hallucinate data when it doesn't have information to fill required properties.

**Example**: If a Person schema requires `name`, `email`, `role`, but content only mentions "Alice works on AI", the LLM fabricates plausible values like `alice@company.com` or "Software Engineer" instead of indicating the information isn't available.

## Solution: Three-Part Strategy

### 1. Make All Properties Nullable

**Approach**: Use JSON Schema **union types** `["type", "null"]` to allow both the original type AND `null`.

✅ **Confirmed**: This is the official Azure/OpenAI documented pattern for nullable properties in structured outputs.

```json
// Before (forces hallucination)
{
  "name": {"type": "string"},
  "email": {"type": "string"}
}

// After (allows null using union type)
{
  "name": {
    "type": ["string", "null"],
    "description": "Person's name. Use null if not available in the content."
  },
  "email": {
    "type": ["string", "null"],
    "description": "Person's email. Use null if not available in the content."
  }
}
```

For enums, use union types `["string", "null"]` with the enum constraint (NOT null in the enum array):
```json
{
  "status": {
    "type": ["string", "null"],
    "enum": ["active", "inactive", "pending"],
    "description": "Status or null if not available. Must be one of: active, inactive, pending, or null if not available."
  }
}
```

**⚠️ CRITICAL**: OpenAI does NOT allow `null` in enum arrays. Adding `null` directly to the enum list causes validation errors:
```json
// ❌ BROKEN - OpenAI rejects this
{
  "status": {
    "enum": ["active", "inactive", null]  // ❌ Validation error!
  }
}

// ✅ CORRECT - Use union type instead
{
  "status": {
    "type": ["string", "null"],  // ✅ Allows null via union type
    "enum": ["active", "inactive"]  // ✅ Enum array has no null
  }
}
```

**How it works**: The union type `["string", "null"]` tells OpenAI the property can be a string OR null, while the enum constrains string values to the allowed list. Combined, this means "must be one of the enum values OR null".

**Note**: We use union types `["type", "null"]` instead of `anyOf: [{type: "string"}, {type: "null"}]` because:
- ✅ Simpler and more readable
- ✅ Explicitly documented by Azure/OpenAI
- ✅ Standard JSON Schema pattern for optional/nullable fields

### 2. Update System Prompts

Added explicit anti-hallucination instructions to the LLM:

```
ANTI-HALLUCINATION RULES:
1. Only extract information that is explicitly present in the provided content
2. For any property where the information is not available, use null instead of guessing
3. It is better to have null properties than incorrect or hallucinated data
4. Do not infer, assume, or create information that is not directly stated

All properties accept null values when information is not available in the content.
```

### 3. Filter Nulls Before Storage

Created `_filter_null_properties()` to remove null values before storing in Neo4j:

```python
def _filter_null_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    """Remove null values recursively from node properties"""
    filtered = {}
    for key, value in properties.items():
        if value is not None:
            if isinstance(value, dict):
                filtered_nested = _filter_null_properties(value)
                if filtered_nested:
                    filtered[key] = filtered_nested
            elif isinstance(value, list):
                filtered_list = [item for item in value if item is not None]
                if filtered_list:
                    filtered[key] = filtered_list
            else:
                filtered[key] = value
    return filtered
```

## Complete Flow

```
┌─────────────────────┐
│  LLM Extraction     │ → Returns nodes with nulls for missing data
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Null Filtering     │ → Removes all null values
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Property Overrides  │ → Can fill missing required fields
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Required Validation │ → Check schema required fields
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Create/Skip Node   │ → Store if valid, skip if missing required
└─────────────────────┘
```

## Example: Before & After

### Before (Hallucination)
```json
// Content: "Alice works on Project Phoenix"

// LLM Output - FABRICATES missing data
{
  "label": "Person",
  "properties": {
    "name": "Alice",
    "email": "alice@company.com",     // ❌ HALLUCINATED
    "role": "Software Engineer",       // ❌ HALLUCINATED
    "phone": "+1-555-0123"            // ❌ HALLUCINATED
  }
}

// Stored in Neo4j - mixed real/fake data ❌
```

### After (Truthful)
```json
// Content: "Alice works on Project Phoenix"

// LLM Output - HONEST about missing data
{
  "label": "Person",
  "properties": {
    "name": "Alice",
    "email": null,     // ✅ Honest: not available
    "role": null,      // ✅ Honest: not available
    "phone": null      // ✅ Honest: not available
  }
}

// After null filtering - only real data ✅
{
  "label": "Person",
  "properties": {
    "name": "Alice"    // ✅ Only actual extracted data
  }
}
```

## Important Distinctions

### Two Types of "Required"

1. **OpenAI Required (Technical Constraint)**
   - ALL properties must be in `required` array
   - API requirement for structured outputs
   - **Solution**: Make properties nullable with `anyOf`

2. **Schema Required (Business Logic)**
   - Properties that MUST have values for node to be valid
   - Defined in `UserNodeType.required_properties`
   - **Solution**: Validate after filtering, skip node if missing

### Property Overrides Integration

Property overrides work seamlessly with null filtering:

```python
# LLM output (missing required field)
{"name": "Alice", "email": null}  # email is required

# After null filtering
{"name": "Alice"}  # email missing

# Property override applied
property_override = {
    "nodeLabel": "Person",
    "match": {"name": "Alice"},
    "set": {"email": "alice@company.com", "id": "person_alice_123"}
}

# Final result - VALID! ✅
{"name": "Alice", "email": "alice@company.com", "id": "person_alice_123"}
```

**Key Point**: Property overrides can "rescue" nodes by providing missing required fields.

### Dynamic Schemas Support

✅ **Works with all schema types**:
- System schemas (Memory, Person, Company, etc.)
- User-defined custom schemas
- Runtime/dynamic schemas (nodes not known in advance)

The nullable transformation is applied programmatically at runtime, regardless of schema source.

## Implementation Details

### Files Modified

1. **`memory/memory_graph.py`**
   - `_filter_null_properties()` - Remove null values from properties
   - `_validate_required_properties()` - Validate schema required fields
   - `_make_schema_nullable()` - Transform schemas to accept nulls
   - `get_custom_schema_for_structured_output()` - Generate nullable custom schemas
   - `get_memory_graph_schema()` - Apply nullable to system schema
   - `get_node_schema()` - Apply nullable to node schema
   - `_create_node()` - Apply null filtering before creation
   - `_merge_node_with_unique_identifiers()` - Apply null filtering before merge

2. **`api_handlers/chat_gpt_completion.py`**
   - Updated system prompt in `generate_memory_graph_schema_async()`
   - Added validation in `generate_node_ids()` after property overrides

### Key Methods

#### Filter Null Properties
```python
@staticmethod
def _filter_null_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    """Remove null values to prevent storing hallucinated data"""
    # Recursively filters nulls from dicts, lists, and nested structures
```

#### Validate Required Properties
```python
@staticmethod
def _validate_required_properties(
    node_label: str,
    properties: Dict[str, Any],
    user_schema: Optional[Any] = None
) -> tuple[bool, List[str]]:
    """Check schema-required fields are present after filtering"""
    # Returns (is_valid, missing_fields)
```

#### Make Schema Nullable
```python
@staticmethod
def _make_schema_nullable(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively transform schema to accept nulls"""
    # Converts {"type": "string"} → {"type": ["string", "null"]}
```

## Validation Logic

```python
# In generate_node_ids(), after property overrides:
if user_schema:
    is_valid, missing_fields = MemoryGraph._validate_required_properties(
        node['label'], final_properties, user_schema
    )
    if not is_valid:
        logger.warning(f"🚫 SKIPPING {node['label']} - missing: {missing_fields}")
        continue  # Don't create node
```

## Testing

### Test Cases

1. **Partial Information** - Only some properties available
2. **All Required Present** - Node should be created
3. **Missing Required** - Node should be skipped with warning
4. **Property Override Rescue** - Override provides missing required field

### Verification Queries

```cypher
// Find potentially hallucinated data (should return empty)
MATCH (p:Person)
WHERE p.email IS NOT NULL 
  AND NOT exists((:Memory)-[:MENTIONS]->(p))
RETURN p.email

// Verify only real properties stored
MATCH (n)
RETURN labels(n), keys(n), count(*) as node_count
ORDER BY node_count DESC
```

### Log Indicators

**Success**:
```
🚫 ANTI-HALLUCINATION: Filtered null properties. Before: 7, After: 3
✓ VALIDATION: Person has all required properties: ['id', 'name']
🔧 APPLIED OVERRIDES for Person (id: person-123): {'email': 'alice@company.com'}
```

**Expected Warnings** (for incomplete data):
```
✗ VALIDATION: Developer missing required properties: ['email']
🚫 SKIPPING Developer node (id: dev-456) - missing required fields: ['email']
```

## Benefits

| Aspect | Before | After |
|--------|--------|-------|
| **Data Accuracy** | ❌ Mixed real/fake | ✅ Only real data |
| **OpenAI Compliance** | ⚠️ Workaround | ✅ Fully compliant |
| **Required Fields** | ❌ Not validated | ✅ Validated |
| **Dynamic Schemas** | ⚠️ Limited | ✅ Full support |
| **Property Overrides** | ✅ Working | ✅ Enhanced |
| **Hallucination** | ❌ Common | ✅ Prevented |

## Backward Compatibility

✅ **Fully backward compatible**:
- No API changes required
- No database migrations needed
- Existing memories unaffected
- Works with existing client code
- No configuration changes

## Performance Impact

**Negligible**:
- Null filtering: O(n) where n = number of properties
- Schema transformation: One-time cost at generation
- Validation: O(m) where m = number of required properties
- No additional LLM API calls

## Edge Cases Handled

✅ Nested properties with nulls - Recursively filtered  
✅ Arrays with null items - Null items removed  
✅ Enum properties - Null allowed via union types `["string", "null"]` (NOT in enum array - OpenAI validation fix)  
✅ Empty objects after filtering - Removed entirely  
✅ Property overrides rescuing nodes - Validated after overrides  
✅ System vs custom schemas - Both handled  
✅ Dynamic runtime schemas - Supported  
✅ Missing schema definition - Gracefully skipped  
✅ Mixed required/optional fields - Correctly distinguished  

## Configuration

**No configuration needed!** The fix is automatic for:
- All custom user schemas
- All system schemas  
- All add memory operations
- All node types

## Future Enhancements

Possible improvements:
1. **Confidence Scores** - Add confidence level for each property
2. **Uncertainty Indicators** - Distinguish "unknown" from "uncertain"
3. **Property Importance** - Mark properties as critical vs optional
4. **Relationship Validation** - Skip relationships to skipped nodes
5. **LLM Feedback Loop** - Inform LLM which properties are required

## Summary

This solution prevents LLM hallucination while maintaining full compliance with OpenAI's structured output requirements:

1. ✅ **Properties are nullable** - LLM can use `null` for missing data
2. ✅ **System prompts updated** - Explicit instructions against hallucination
3. ✅ **Nulls filtered** - Only real data stored in database
4. ✅ **Required fields validated** - Nodes skipped if missing required fields
5. ✅ **Property overrides work** - Can fill missing required fields
6. ✅ **Dynamic schemas supported** - Works with any schema type
7. ✅ **Fully backward compatible** - No breaking changes

**⚠️ IMPORTANT FIX (November 2024)**: OpenAI does NOT allow `None` in enum arrays. We use union types `["string", "null"]` instead:
- ❌ **BROKEN**: `{"enum": ["value1", "value2", None]}` - OpenAI rejects this
- ✅ **CORRECT**: `{"type": ["string", "null"], "enum": ["value1", "value2"]}` - OpenAI accepts this

**Status**: ✅ Production Ready  
**Last Updated**: November 2024  
**Breaking Changes**: None  
**Migration Required**: None

