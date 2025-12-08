# Delete All Memories Tests

This document describes the comprehensive test suite for the new `DELETE /v1/memory/all` endpoint.

## Overview

The `delete_all_memories` endpoint allows developers to delete all memories for a specific user. These tests verify that the endpoint works correctly across different scenarios and user resolution methods.

## Test Files

- **`test_delete_all_memories.py`** - Main test file containing all test cases
- **`run_delete_all_memories_tests.py`** - Standalone test runner for development

## Test Cases

### 1. Complete Workflow Test (`test_delete_all_memories_complete_workflow`)

**Purpose**: End-to-end test that validates the entire delete_all_memories workflow.

**Steps**:
1. 📝 Creates a new test user with unique credentials
2. 📚 Adds 5 memories via batch operation for that user
3. 🔍 Verifies memories exist by searching for them
4. 🗑️ Deletes ALL memories for the user using `DELETE /v1/memory/all`
5. 🔍 Confirms memories were deleted by searching again
6. 🔍 Additional verification by checking specific memory IDs

**Verification**:
- Batch creation of 5 memories succeeds
- Search finds the created memories before deletion
- Delete all endpoint processes and deletes all memories
- Search returns no memories after deletion
- Individual memory GET requests return 404

### 2. External User ID Test (`test_delete_all_memories_with_external_user_id`)

**Purpose**: Tests user resolution via `external_user_id` parameter.

**Steps**:
1. 📝 Creates a test user with an external_user_id
2. 📚 Adds a memory for that user
3. 🗑️ Deletes all memories using `external_user_id` parameter
4. ✅ Verifies deletion succeeded

**Key Feature**: Tests that the endpoint correctly resolves `external_user_id` to internal user ID.

### 3. No Memories Found Test (`test_delete_all_memories_no_memories_found`)

**Purpose**: Tests edge case where user has no memories to delete.

**Steps**:
1. 📝 Creates a new test user
2. 🗑️ Attempts to delete all memories (user has none)
3. ✅ Verifies appropriate 404 response with correct error message

**Expected Behavior**: Returns 404 with "No memories found for user" message.

## Running the Tests

### Option 1: Standalone Test Runner

```bash
# Run just the delete_all_memories tests
cd /path/to/memory
python tests/run_delete_all_memories_tests.py
```

### Option 2: Sequential Test Runner

```bash
# Run all v1 endpoint tests (includes delete_all_memories tests)
python tests/test_v1_endpoints_sequential.py
```

### Option 3: Pytest

```bash
# Run specific test file
pytest tests/test_delete_all_memories.py -v

# Run specific test function
pytest tests/test_delete_all_memories.py::test_delete_all_memories_complete_workflow -v
```

## Test Data

Each test uses unique identifiers to avoid conflicts:

- **Test Run ID**: 12-character UUID hex for uniqueness
- **Test Email**: `delete_test_user_{run_id}@example.com`
- **External User ID**: `delete_test_user_{run_id}`
- **Memory Content**: Includes run_id for traceability

## Environment Requirements

- **TEST_X_USER_API_KEY**: Required environment variable for API authentication
- **Parse Server**: Must be running and accessible
- **Qdrant**: Vector database must be accessible
- **Neo4j**: Graph database must be accessible

## Expected Output

### Successful Test Run

```
🧪 Starting delete_all_memories test with run ID: abc123def456
📝 Step 1: Creating new test user...
✅ Created test user with ID: GfZhDFxnS6
📚 Step 2: Adding batch memories for the test user...
✅ Successfully created 5 memories for user GfZhDFxnS6
🔍 Step 3: Verifying memories exist via search...
🔍 Found 5 memories in search before deletion
🗑️ Step 4: Deleting ALL memories for the test user...
✅ Delete all results:
   - Total processed: 5
   - Total successful: 5
   - Total failed: 0
🔍 Step 5: Confirming memories are deleted via search...
✅ Search returned 404 - no memories found (as expected)
🔍 Step 6: Additional verification - checking specific memory IDs...
✅ Memory mem_abc123_1 correctly returns 404 (deleted)
✅ Memory mem_abc123_2 correctly returns 404 (deleted)
✅ Memory mem_abc123_3 correctly returns 404 (deleted)
📊 Verification summary: 3/3 checked memories are properly deleted
🏁 Final verification...
✅ All deletions successful - test completed perfectly!
🎉 delete_all_memories test completed successfully for run ID: abc123def456
```

## API Endpoint Usage Examples

### Delete all memories for developer (API key holder)
```bash
curl -X DELETE "https://api.papr.ai/v1/memory/all" \
  -H "X-API-Key: your_api_key" \
  -H "X-Client-Type: papr_plugin"
```

### Delete all memories for specific user
```bash
curl -X DELETE "https://api.papr.ai/v1/memory/all?user_id=user123" \
  -H "X-API-Key: your_api_key" \
  -H "X-Client-Type: papr_plugin"
```

### Delete all memories for external user
```bash
curl -X DELETE "https://api.papr.ai/v1/memory/all?external_user_id=ext_user456" \
  -H "X-API-Key: your_api_key" \
  -H "X-Client-Type: papr_plugin"
```

## Response Format

### Successful Deletion (200)
```json
{
  "code": 200,
  "status": "success",
  "data": [],
  "error": null,
  "errors": [],
  "total_processed": 5,
  "total_successful": 5,
  "total_failed": 0,
  "details": {
    "message": "Successfully deleted all 5 memories for user abc123"
  }
}
```

### Partial Success (207)
```json
{
  "code": 207,
  "status": "partial_success",
  "data": [],
  "error": "Partial success: 3 deleted, 2 failed",
  "errors": [
    {
      "index": 3,
      "error": "Failed to delete memory mem_xyz: Connection timeout"
    }
  ],
  "total_processed": 5,
  "total_successful": 3,
  "total_failed": 2
}
```

### No Memories Found (404)
```json
{
  "code": 404,
  "status": "error",
  "data": [],
  "error": "No memories found for user",
  "errors": [
    {
      "index": -1,
      "error": "No memories found for user"
    }
  ]
}
```

## Security & Safety Features

- ⚠️ **Consequential Operation**: Marked as `x-openai-isConsequential: True`
- 🔐 **Authentication Required**: API key, Bearer token, or Session token
- 👤 **User Resolution**: Proper user ID resolution via optimized auth
- 📊 **Detailed Logging**: Comprehensive audit trail
- 📈 **Analytics Tracking**: Amplitude events for monitoring
- 🔄 **Pagination Support**: Handles users with >1000 memories
- ⚡ **Concurrent Processing**: Optimized batch deletion with limits
- 🛡️ **Error Handling**: Graceful handling of partial failures

## Troubleshooting

### Common Issues

1. **Environment Variables Missing**
   - Ensure `TEST_X_USER_API_KEY` is set
   - Check that all database connections are configured

2. **Connection Timeouts**
   - Verify Parse Server is accessible
   - Check Qdrant and Neo4j connectivity

3. **Memory Creation Failures**
   - Verify user has proper permissions
   - Check API key validity

4. **Search Results Inconsistent**
   - Allow time for indexing (tests include delays)
   - Check vector database synchronization

### Debug Mode

Run tests with detailed logging:
```bash
PYTHONPATH=. python -m pytest tests/test_delete_all_memories.py -v -s --log-cli-level=INFO
``` 