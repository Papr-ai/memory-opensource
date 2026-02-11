# Document Upload with Custom Schema

## Overview

The `/v1/document/v2` endpoint now properly validates metadata using the `UploadDocumentRequest` Pydantic model, which includes support for `schema_id` to enforce custom graph schemas during memory extraction.

## Request Format

### Metadata Structure

The `metadata` field in the upload request should be a JSON string with the following structure:

```json
{
  "schema_id": "Dh6EivRmo8",  // Optional: Custom schema ID for graph enforcement
  "metadata": {               // Optional: MemoryMetadata fields
    "source": "test_upload",
    "domain": "security",
    "tags": ["two-factor-auth", "security"],
    "customMetadata": {
      "department": "engineering",
      "project": "auth_security"
    }
  }
}
```

### Pydantic Model

```python
class UploadDocumentRequest(BaseModel):
    """Request model for uploading a document with metadata."""
    type: MemoryType = Field(
        default=MemoryType.DOCUMENT,
        description="Type of memory. Only 'document' is allowed."
    )
    metadata: Optional[MemoryMetadata] = Field(
        None,
        description="Metadata for the document upload, including user and ACL fields."
    )
    schema_id: Optional[str] = Field(
        None,
        description="Optional custom schema ID to enforce specific node/relationship types during memory extraction."
    )
```

## Example Uploads

### 1. With Custom Schema (Security Behaviors & Risk)

```bash
curl -X POST http://localhost:8000/v1/document/v2 \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@two-factor_authentication.pdf" \
  -F 'metadata={"schema_id":"Dh6EivRmo8","metadata":{"source":"security_docs","domain":"security","tags":["two-factor","authentication"]}}' \
  -F "preferred_provider=reducto" \
  -F "hierarchical_enabled=true"
```

**Expected Neo4j Node Types:**
- `SecurityBehavior`
- `Control`
- `RiskIndicator`
- `Impact`
- `VerificationMethod`

### 2. With Custom Schema (Customer Support & Workflows)

```bash
curl -X POST http://localhost:8000/v1/document/v2 \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@support_transcript.pdf" \
  -F 'metadata={"schema_id":"ABC123XYZ","metadata":{"source":"support_docs","domain":"customer_support"}}' \
  -F "preferred_provider=reducto" \
  -F "hierarchical_enabled=true"
```

**Expected Neo4j Node Types:**
- `CallSession`
- `Utterance`
- `Workflow`
- `Step`
- `WorkflowRun`
- `Agent`
- `Customer`

### 3. Without Custom Schema (System Default)

```bash
curl -X POST http://localhost:8000/v1/document/v2 \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@document.pdf" \
  -F 'metadata={"metadata":{"source":"general_docs"}}' \
  -F "preferred_provider=reducto" \
  -F "hierarchical_enabled=true"
```

**Expected Neo4j Node Types (System Schema):**
- `Memory`
- `Goal`
- `UseCase`
- `Person`
- `Company`
- `Project`
- `Task`
- `Insight`

### 4. Minimal Upload (No Metadata)

```bash
curl -X POST http://localhost:8000/v1/document/v2 \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@document.pdf" \
  -F "preferred_provider=reducto" \
  -F "hierarchical_enabled=true"
```

## Schema ID Flow

```
User Upload Request
  └─> metadata JSON string
      └─> Parse as UploadDocumentRequest (Pydantic validation)
          ├─> schema_id: Optional[str]
          └─> metadata: Optional[MemoryMetadata]
              └─> Apply multi-tenant scoping
                  └─> Enrich with file info (file_url, upload_id, user_id)
                      └─> Pass to DocumentProcessingWorkflow
                          └─> Store memories with schema_id in customMetadata
                              └─> Start child workflow (ProcessBatchMemoryFromPostWorkflow)
                                  └─> Fetch memories with schema_id
                                      └─> LLM enforces custom schema
                                          └─> Neo4j nodes created with custom types
```

## Validation Benefits

1. **✅ Type Safety**: Pydantic validates all fields at request time
2. **✅ Clear Errors**: Get meaningful error messages for invalid metadata
3. **✅ Auto Documentation**: OpenAPI spec automatically generated
4. **✅ IDE Support**: Type hints work in IDEs
5. **✅ Consistent**: Same model used across document endpoints

## Error Handling

### Invalid JSON
```json
{
  "status": "failure",
  "error": "Invalid metadata JSON",
  "message": "Metadata must be valid JSON: Expecting value: line 1 column 1 (char 0)"
}
```

### Invalid Metadata Structure
```json
{
  "status": "failure",
  "error": "Invalid metadata format",
  "message": "Failed to parse metadata: 1 validation error for UploadDocumentRequest..."
}
```

### Missing Authentication
```json
{
  "status": "failure",
  "error": "Missing authentication",
  "code": 401,
  "message": "Authorization header, X-API-Key, or X-Session-Token required"
}
```

## Metadata Enrichment

The system automatically enriches metadata with:

```python
{
  "customMetadata": {
    "file_url": "https://parse-server.com/files/...",
    "file_name": "document.pdf",
    "upload_id": "uuid-here",
    # ... user's custom metadata ...
  },
  "user_id": "developer_user_id",
  "external_user_id": "end_user_id",
  # ... multi-tenant scoping ...
}
```

## Testing

### 1. Verify Request Parsing
```python
import json

metadata = {
    "schema_id": "Dh6EivRmo8",
    "metadata": {
        "source": "test",
        "domain": "security"
    }
}

# Send as JSON string
metadata_str = json.dumps(metadata)
```

### 2. Check Logs
```bash
# Document worker logs
tail -f .document_worker.out | grep "📋 Parsed upload request"

# Expected output:
# 📋 Parsed upload request - schema_id: Dh6EivRmo8, metadata: source='test' domain='security' ...
# 🚀 Starting DocumentProcessingWorkflow with schema_id: Dh6EivRmo8
```

### 3. Verify in Temporal UI
- Check workflow args include `schema_id`
- Verify child workflow receives `schema_id`
- Confirm activities log schema enforcement

### 4. Query Neo4j
```cypher
// Find memories with custom schema
MATCH (n)
WHERE n.upload_id = '<your_upload_id>'
AND EXISTS(n.schema_id)
RETURN DISTINCT labels(n) as NodeTypes, n.schema_id, count(*) as Count
ORDER BY Count DESC

// Expected: Custom schema node types, not system defaults
```

## Common Issues

### Issue: schema_id Not Passed to Workflow
**Symptom**: Neo4j shows system node types (Memory, Goal, etc.) instead of custom types

**Solution**: Check metadata format:
```json
{
  "schema_id": "Dh6EivRmo8",  // ✅ Correct: at top level
  "metadata": {
    "customMetadata": {
      "schema_id": "Dh6EivRmo8"  // ❌ Wrong: nested in customMetadata
    }
  }
}
```

### Issue: Validation Error
**Symptom**: 400 error with Pydantic validation message

**Solution**: Ensure metadata follows MemoryMetadata structure:
```json
{
  "schema_id": "ABC123",
  "metadata": {
    "source": "string",      // ✅ Correct
    "domain": "string",      // ✅ Correct
    "tags": ["array"],       // ✅ Correct
    "invalid_field": "..."   // ❌ Will be ignored (extra fields allowed)
  }
}
```

### Issue: Empty Metadata
**Symptom**: No error but metadata is empty

**Solution**: System creates empty MemoryMetadata with multi-tenant scoping if no metadata provided

## Python SDK Example

```python
import requests
import json

api_key = "YOUR_API_KEY"
base_url = "http://localhost:8000"

# Prepare metadata
metadata = {
    "schema_id": "Dh6EivRmo8",  # Security Behaviors & Risk schema
    "metadata": {
        "source": "security_analysis",
        "domain": "security",
        "tags": ["two-factor", "authentication", "security"],
        "customMetadata": {
            "department": "engineering",
            "project": "auth_improvements"
        }
    }
}

# Upload document
with open("two-factor_authentication.pdf", "rb") as f:
    files = {"file": f}
    data = {
        "metadata": json.dumps(metadata),
        "preferred_provider": "reducto",
        "hierarchical_enabled": "true"
    }
    headers = {"X-API-Key": api_key}
    
    response = requests.post(
        f"{base_url}/v1/document/v2",
        files=files,
        data=data,
        headers=headers
    )
    
    print(response.json())
```

## References

- `models/shared_types.py` - `UploadDocumentRequest` and `MemoryMetadata` models
- `routers/v1/document_routes_v2.py` - Document upload endpoint
- `cloud_plugins/temporal/workflows/document_processing.py` - Document workflow
- `cloud_plugins/temporal/workflows/batch_memory.py` - Batch memory workflow

