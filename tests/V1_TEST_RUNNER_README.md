# V1 Endpoints Sequential Test Runner

This test runner executes all v1 endpoint tests sequentially to avoid parallel execution issues and provides detailed logging and reporting.

## Overview

The sequential test runner imports all v1 test functions from `tests/test_add_memory_fastapi.py` and `tests/test_user_v1_integration.py` and runs them one by one, providing:

- **Sequential execution** to avoid parallel pytest issues
- **Detailed logging** for each test
- **Comprehensive reporting** in JSON and text formats
- **Test categorization** by endpoint groups
- **Success/failure tracking** with timing information

## Test Categories

The runner organizes tests into the following categories:

### Memory Endpoints

#### 1. Add Memory Tests
- Basic add memory functionality
- API key authentication
- External user ID with custom metadata
- External user ID only
- External user ID with ACL

#### 2. Batch Add Memory Tests
- Basic batch memory addition
- Batch with user ID
- Batch with external user ID
- Webhook immediate when skip_background_processing=True
- Webhook with background processing

#### 3. Update Memory Tests
- Basic memory updates
- API key authentication
- ACL updates with real users

#### 4. Get Memory Tests
- Basic memory retrieval

#### 5. Search Memory Tests
- Basic search functionality
- User ID ACL validation
- External user ID ACL validation
- New user Qwen route testing
- Fixed user cache testing
- Custom metadata filter (Qwen-only)
- Numeric custom metadata filter
- List custom metadata filter
- Boolean custom metadata filter
- Mixed custom metadata types
- Auth/Dev marking: APIKey marks developer
- Auth/Dev marking: Bearer-only does not mark developer
- Low-similarity performance search (adds long-form Papr content for a fixed user and verifies retrieval with a shorter query)

#### 6. Delete Memory Tests
- Basic memory deletion
- API key authentication

#### 7. Upload Document Tests
- Document upload with API key
- Document upload with session token
- Real PDF file upload
- Real PDF with custom schema
- Invalid file type handling
- Malicious content detection
- Large file handling
- Webhook integration
- Multi-tenant document processing
- Provider preference handling
- Authentication failure
- Document status endpoint
- Document cancel endpoint
- Reducto provider (direct and route)
- Reducto provider (simple upload)
- Gemini provider upload
- TensorLake provider upload
- PaddleOCR provider upload
- DeepSeek-OCR provider upload

### User Endpoints

#### 8. Create User Tests
- Basic user creation with email
- Anonymous user creation (no email)

#### 9. Get User Tests
- Basic user retrieval by user ID

#### 10. Update User Tests
- Basic user information updates

#### 11. Delete User Tests
- Basic user deletion by user ID
- User deletion by external ID

#### 12. List Users Tests
- Basic user listing with pagination

### Feedback Endpoints

#### 13. Feedback Tests
- End-to-end feedback submission (search → feedback correlation)

### Query Log Integration

#### 14. Query Log Tests
- QueryLog - Real creation and memory increment (`test_real_query_log_creation_and_memory_increment`)
- QueryLog - Cache hits increment on repeated search (`test_cache_hits_increment_on_repeated_search`)
- QueryLog - Fused confidence matches weight delta (`test_fused_confidence_matches_weight_delta`)

These validate retrieval logging, atomic increments on cache hits, feedback-driven citation updates, and the fused-confidence weighting with time decay as defined in `docs/confidence_weighting_proof.md`.

### Custom Schema Integration

#### 15. Security Schema Tests
-Custom Schema - Create Security Schema (`test_v1_create_security_schema`): Creates a comprehensive security monitoring schema with 12 node types and 20 relationship types
- Custom Schema - Add Memory with schema_id (`test_v1_add_memory_with_schema_id`): Adds memory with schema_id in metadata to trigger LLM schema selection
- Custom Schema - Wait for Memory Processing (`test_v1_wait_for_memory_processing`): Validates background processing completes within 120 seconds
- Custom Schema - Search Verify Neo4j Nodes (`test_v1_search_verify_neo4j_nodes`): Searches for created memory and verifies Neo4j storage
- Custom Schema - Search with Agentic Graph (`test_v1_search_with_agentic_graph`): Tests 2-hop pattern matching with agentic graph enabled
- Custom Schema - Add Memory with graph_override (`test_v1_add_memory_with_graph_override`): Provides pre-made graph structure bypassing LLM generation
- Custom Schema - Full Workflow Validation (`test_v1_security_schema_full_workflow`): End-to-end validation that all components work together

These tests validate the custom schema functionality including schema creation, LLM-powered schema selection, graph_override, background processing, Neo4j storage, and agentic graph search with 2-hop patterns.

## Usage

### Quick Start
```bash
python run_v1_tests.py
```

### Direct Execution
```bash
python test_v1_endpoints_sequential.py
```

### With Custom Logging
```bash
python -u test_v1_endpoints_sequential.py 2>&1 | tee test_run.log
```

## Output

### Console Output
The runner provides real-time feedback:
```
🚀 Starting V1 Endpoints Sequential Test Suite
🧪 Running Add Memory Tests...
Starting test: Add Memory - Basic
✅ Test passed: Add Memory - Basic (took 2.34s)
...
📊 TEST SUMMARY
================================================================================
Total Tests: 18
Passed: 15 ✅
Failed: 3 ❌
Success Rate: 83.3%
Total Duration: 45.67s
Average Duration: 2.54s
```

### Report Files
The runner generates two types of reports in the `tests/test_reports/` directory:

1. **JSON Report** (`v1_endpoints_report_YYYYMMDD_HHMMSS.json`)
   - Machine-readable format
   - Complete test results with timing
   - Error details for failed tests
   - Summary statistics

2. **Text Report** (`v1_endpoints_log_YYYYMMDD_HHMMSS.txt`)
   - Human-readable format
   - Test-by-test results
   - Error messages for failed tests
   - Summary statistics

## Test Functions

The runner imports test functions from two files:

### Memory Endpoints (`tests/test_add_memory_fastapi.py`)

#### Add Memory Tests
- `test_v1_add_memory_1`
- `test_v1_add_memory_with_api_key`
- `test_v1_add_memory_with_external_user_id_and_custom_metadata`
- `test_v1_add_memory_with_external_user_id_only`
- `test_v1_add_memory_with_external_user_id_and_acl`

#### Batch Add Memory Tests
- `test_v1_add_memory_batch_1`
- `test_v1_add_memory_batch_with_user_id`
- `test_v1_add_memory_batch_with_external_user_id`

#### Update Memory Tests
- `test_v1_update_memory_1`
- `test_v1_update_memory_with_api_key`
- `test_v1_update_memory_acl_with_api_key_and_real_users`

#### Get Memory Tests
- `test_v1_get_memory`

#### Search Memory Tests
- `test_v1_search_1`
- `test_v1_search_with_user_id_acl`
- `test_v1_search_with_external_user_id_acl`
- `test_v1_search_new_user_qwen_route`
- `test_v1_search_fixed_user_cache_test`

#### Delete Memory Tests
- `test_v1_delete_memory_1`
- `test_v1_delete_memory_with_api_key`

#### Upload Document Tests
- `test_document_upload_v2_with_api_key`
- `test_document_upload_v2_with_session_token`
- `test_document_upload_v2_with_real_pdf_file`
- `test_document_upload_v2_with_real_pdf_file_custom_schema`
- `test_document_upload_v2_invalid_file_type`
- `test_document_upload_v2_malicious_content`
- `test_document_upload_v2_large_file`
- `test_document_upload_v2_with_webhook`
- `test_document_upload_v2_multi_tenant`
- `test_document_upload_v2_provider_preference`
- `test_document_upload_v2_authentication_failure`
- `test_document_status_endpoint`
- `test_document_cancel_endpoint`
- `test_reducto_provider_direct_and_route`
- `test_reducto_provider_simple_upload_and_parse`
- `test_document_upload_v2_with_gemini_provider`
- `test_document_upload_v2_with_tensorlake_provider`
- `test_document_upload_v2_with_paddleocr_provider`
- `test_document_upload_v2_with_deepseek_ocr_provider`

### User Endpoints (`tests/test_user_v1_integration.py`)

#### Create User Tests
- `test_create_user_v1_integration`
- `test_create_anonymous_user_v1_integration`

#### Get User Tests
- `test_get_user_v1_integration`

#### Update User Tests
- `test_update_user_v1_integration`

#### Delete User Tests
- `test_delete_user_v1_integration`
- `test_delete_user_by_external_id_integration`

#### List Users Tests
- `test_list_users_v1_integration`

### Feedback Endpoints (`tests/test_feedback_end_to_end.py`)

#### Feedback Tests
- `test_feedback_end_to_end`

## Benefits

1. **Avoids Parallel Issues**: Sequential execution prevents pytest parallel execution problems
2. **Detailed Logging**: Each test is logged with timing and status
3. **Comprehensive Reporting**: Both JSON and text reports for different use cases
4. **Easy Debugging**: Clear error messages and test categorization
5. **CI/CD Ready**: Structured output suitable for CI/CD pipelines

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all test functions exist in `tests/test_add_memory_fastapi.py`
2. **App Startup Timeout**: Increase `startup_timeout` in `LifespanManager` if needed
3. **Test Dependencies**: Some tests may depend on external services (Neo4j, Pinecone, etc.)

### Adding New Tests

To add new v1 test functions:

1. Add the test function to the appropriate test file:
   - Memory endpoints: `tests/test_add_memory_fastapi.py`
   - User endpoints: `tests/test_user_v1_integration.py`
   - Feedback endpoints: `tests/test_feedback_end_to_end.py`
2. Import it in `test_v1_endpoints_sequential.py`
3. Add it to the appropriate test category method
4. Update this README with the new test

### Customization

You can modify the test runner to:
- Add new test categories
- Change logging levels
- Modify report formats
- Add custom test validation

## Example Report Structure

```json
{
  "summary": {
    "total_tests": 19,
    "passed_tests": 16,
    "failed_tests": 3,
    "success_rate": 84.2,
    "total_duration": 48.01,
    "average_duration": 2.53
  },
  "results": [
    {
      "test_name": "Add Memory - Basic",
      "status": "passed",
      "duration": 2.34,
      "error": null
    }
  ],
  "timestamp": "2024-01-17T17:30:45.123456"
}
```

## Integration with CI/CD

The JSON report format makes it easy to integrate with CI/CD systems:

```bash
# Run tests and capture exit code
python test_v1_endpoints_sequential.py
EXIT_CODE=$?

# Parse results for CI/CD
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed"
else
    echo "❌ Some tests failed"
    # Parse JSON report for detailed failure info
    python -c "
import json
with open('test_reports/v1_endpoints_report_*.json') as f:
    data = json.load(f)
    failed = [r for r in data['results'] if r['status'] == 'failed']
    for test in failed:
        print(f'Failed: {test[\"test_name\"]} - {test[\"error\"]}')
    "
fi
``` 