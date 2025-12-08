# Custom Schema API Implementation

## Overview

This implementation adds comprehensive support for custom developer schemas in API responses, following best practices for API documentation and developer experience.

## 🎯 **Implementation Approach: Dynamic Schema Reference**

We chose the **Dynamic Schema Reference** approach because it provides:
- Clean API responses without bloated inline schema definitions
- Discoverable schemas via dedicated endpoints
- Consistent response structure for all node types
- Efficient responses with minimal overhead
- Self-documenting API with clear schema references

## 📋 **Changes Made**

### 1. **Updated Node Model** (`models/memory_models.py`)

```python
class Node(BaseModel):
    """Public-facing node structure - supports both system and custom schema nodes"""
    label: str = Field(..., description="Node type label - can be system type (Memory, Person, etc.) or custom type from UserGraphSchema")
    properties: Dict[str, Any] = Field(..., description="Node properties - structure depends on node type and schema")
    schema_id: Optional[str] = Field(
        None, 
        description="Reference to UserGraphSchema ID for custom nodes. Use GET /v1/schemas/{schema_id} to get full schema definition. Null for system nodes."
    )
```

**Key Features:**
- Flexible `label` field accepts both system and custom node types
- Generic `properties` field supports any schema structure
- `schema_id` field provides reference to UserGraphSchema for custom nodes
- Comprehensive examples in OpenAPI documentation

### 2. **Enhanced SearchResult Model** (`models/memory_models.py`)

```python
class SearchResult(BaseModel):
    """Return type for SearchResult"""
    memories: List[Memory]
    nodes: List[Node]
    schemas_used: Optional[List[str]] = Field(
        None,
        description="List of UserGraphSchema IDs used in this response. Use GET /v1/schemas/{id} to get full schema definitions."
    )
```

**Key Features:**
- New `schemas_used` field lists all schema IDs in the response
- Updated `from_internal` method with schema mapping support
- Automatic detection of custom vs. system nodes

### 3. **Schema Mapping Logic** (`routers/v1/memory_routes_v1.py`)

```python
# Create schema mapping for custom nodes
schema_mapping = {}
try:
    from services.schema_service import SchemaService
    schema_service = SchemaService()
    user_schemas = await schema_service.get_active_schemas(resolved_user_id, workspace_id)
    
    # Build mapping from node label to schema ID
    for schema in user_schemas:
        for node_type in schema.node_types.values():
            schema_mapping[node_type.name] = schema.id
            
    logger.info(f"🔗 SCHEMA MAPPING: Built mapping for {len(schema_mapping)} custom node types from {len(user_schemas)} schemas")
    
except Exception as e:
    logger.warning(f"Failed to build schema mapping: {e}")
    schema_mapping = {}

# Convert with schema mapping
search_result = SearchResult.from_internal(relevant_items, schema_mapping=schema_mapping)
```

**Key Features:**
- Fetches active user schemas during search operations
- Builds mapping from node labels to schema IDs
- Graceful fallback if schema fetching fails
- Comprehensive logging for debugging

### 4. **Enhanced API Documentation** (`routers/v1/memory_routes_v1.py`)

Added comprehensive OpenAPI documentation with:

**Response Examples:**
- System nodes only response
- Custom schema nodes response  
- Mixed system and custom nodes response

**Documentation Sections:**
- Custom Schema Support explanation
- Node type definitions (system vs. custom)
- Schema reference usage instructions
- Agentic graph benefits for custom schemas

## 📊 **API Response Examples**

### System Nodes Only
```json
{
  "code": 200,
  "status": "success",
  "data": {
    "memories": [...],
    "nodes": [
      {
        "label": "Person",
        "properties": {
          "id": "person-123",
          "name": "John Doe",
          "role": "Manager"
        },
        "schema_id": null
      }
    ],
    "schemas_used": null
  }
}
```

### Custom Schema Nodes
```json
{
  "code": 200,
  "status": "success",
  "data": {
    "memories": [...],
    "nodes": [
      {
        "label": "Developer",
        "properties": {
          "id": "dev-789",
          "name": "Rachel Green",
          "expertise": ["React", "TypeScript"],
          "years_experience": 5
        },
        "schema_id": "schema_abc123"
      }
    ],
    "schemas_used": ["schema_abc123"]
  }
}
```

### Mixed Nodes
```json
{
  "code": 200,
  "status": "success",
  "data": {
    "memories": [...],
    "nodes": [
      {
        "label": "Meeting",
        "properties": {...},
        "schema_id": null
      },
      {
        "label": "Product",
        "properties": {...},
        "schema_id": "schema_ecommerce_456"
      }
    ],
    "schemas_used": ["schema_ecommerce_456"]
  }
}
```

## 🔗 **Schema Discovery Workflow**

1. **Search Request** → Returns nodes with `schema_id` references
2. **Schema Discovery** → Use `GET /v1/schemas/{schema_id}` to get full schema definition
3. **Schema Details** → Includes node types, relationships, properties, validation rules

Example schema discovery:
```bash
# 1. Search returns custom nodes
POST /v1/memory/search
# Response includes: "schemas_used": ["schema_abc123"]

# 2. Get schema definition
GET /v1/schemas/schema_abc123
# Response includes full schema with node types, relationships, etc.
```

## 🧪 **Testing**

Created `test_custom_schema_api_response.py` to verify:
- Custom nodes include `schema_id` references
- `schemas_used` field is populated correctly
- Schema endpoint references work properly
- Response structure matches documentation

## 🚀 **Benefits for Developers**

### **API Consumers:**
- **Predictable Structure**: Same response format regardless of schema type
- **Discoverable Schemas**: Clear path to get full schema definitions
- **Type Safety**: Can validate against known schemas
- **Efficient**: No bloated responses with inline schema definitions

### **Schema Creators:**
- **Flexible**: Any custom node types and properties supported
- **Consistent**: Same patterns work for all custom schemas
- **Traceable**: Clear lineage from nodes back to schema definitions
- **Maintainable**: Schema changes don't break API contracts

### **System Benefits:**
- **Scalable**: Works with any number of custom schemas
- **Performance**: Minimal overhead in responses
- **Backward Compatible**: Existing system nodes unchanged
- **Future Proof**: Easy to extend with new schema features

## 📝 **Usage Instructions**

### For API Consumers:
1. Make search requests as usual
2. Check `schemas_used` field in responses
3. For custom nodes, use `schema_id` to fetch full schema definitions
4. Use schema definitions for validation and type safety

### For Schema Creators:
1. Create UserGraphSchema via `POST /v1/schemas`
2. Define custom node types and relationships
3. Search responses automatically include schema references
4. Monitor usage via schema endpoint analytics

## 🔧 **Implementation Notes**

- Schema mapping is built fresh for each search request to ensure accuracy
- Graceful fallback if schema fetching fails (empty schema mapping)
- System nodes always have `schema_id: null`
- Custom nodes always include valid `schema_id` references
- `schemas_used` is sorted list for consistent output
- Comprehensive logging for debugging and monitoring

This implementation provides a robust, scalable foundation for custom schema support while maintaining excellent developer experience and API consistency.
