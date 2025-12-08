# Tests Directory Organization

## Current Structure
```
tests/
├── run_v1_tests.py              # Main test runner entry point
├── test_v1_endpoints_sequential.py  # Sequential test runner implementation
├── conftest.py                  # Pytest configuration
├── __init__.py                  # Package marker
├── 
├── # Endpoint-specific test files
├── test_add_memory_fastapi.py   # Memory endpoint tests
├── test_user_v1_integration.py  # User endpoint tests
├── test_feedback_end_to_end.py  # Feedback endpoint tests
├── test_memory_graph.py         # Memory graph tests
├── test_query_log_integration.py # Query logging tests
├── 
├── # Performance tests
├── test_interaction_limits_*.py
├── 
├── # Test artifacts
├── test_reports/               # Generated test reports
├── *.pdf, *.pkl               # Test data files
└── V1_TEST_RUNNER_README.md   # Test runner documentation
```

## Recommended Organization (Future)

For better organization as your test suite grows:

```
tests/
├── runners/
│   ├── __init__.py
│   ├── run_all_tests.py        # Main entry point
│   ├── run_v1_tests.py         # V1 endpoints runner
│   ├── run_performance_tests.py # Performance test runner
│   └── sequential_runner.py    # Sequential execution logic
│
├── unit/                       # Unit tests
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_services.py
│   └── test_utils.py
│
├── integration/                # Integration tests
│   ├── __init__.py
│   ├── v1/
│   │   ├── __init__.py
│   │   ├── test_memory_routes.py
│   │   ├── test_user_routes.py
│   │   ├── test_feedback_routes.py
│   │   └── test_document_routes.py
│   ├── v2/                     # Future API version
│   └── external/               # External service tests
│       ├── test_stripe_integration.py
│       ├── test_mongodb_integration.py
│       └── test_neo4j_integration.py
│
├── performance/                # Performance tests
│   ├── __init__.py
│   ├── test_memory_search_perf.py
│   ├── test_rate_limiting_perf.py
│   └── test_interaction_limits.py
│
├── e2e/                       # End-to-end tests
│   ├── __init__.py
│   ├── test_full_user_journey.py
│   └── test_feedback_workflow.py
│
├── fixtures/                  # Test data and fixtures
│   ├── __init__.py
│   ├── sample_documents/
│   ├── test_data.json
│   └── mock_responses.py
│
├── reports/                   # Test reports (gitignored)
│   ├── coverage/
│   ├── performance/
│   └── integration/
│
├── conftest.py               # Global pytest configuration
├── __init__.py
└── README.md                 # This file
```

## Migration Strategy

1. **Phase 1 (Current)**: Keep existing structure, use current runners
2. **Phase 2**: Create `runners/` directory, move test runners there
3. **Phase 3**: Organize tests by type (unit/integration/performance)
4. **Phase 4**: Split by API version and create dedicated test suites

## Running Tests

### Current Commands
```bash
# Run all V1 tests sequentially
python tests/run_v1_tests.py

# Run specific test file
python -m pytest tests/test_add_memory_fastapi.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Future Commands (Recommended)
```bash
# Run all tests
python tests/runners/run_all_tests.py

# Run specific test category
python tests/runners/run_v1_tests.py
python tests/runners/run_performance_tests.py

# Run specific test type
python -m pytest tests/unit/ -v
python -m pytest tests/integration/v1/ -v
python -m pytest tests/performance/ -v
```

## Benefits of Recommended Structure

1. **Clear separation** of concerns (unit vs integration vs performance)
2. **Scalable** as you add more API versions
3. **Easier maintenance** with focused test files
4. **Better CI/CD integration** (run different test types in parallel)
5. **Improved test discovery** and organization 