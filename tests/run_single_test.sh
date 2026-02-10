#!/bin/bash
# Script to run a single test in the Docker test-runner container
# Usage: ./run_single_test.sh "test_module::test_function"

if [ -z "$1" ]; then
    echo "Usage: $0 <test_path>"
    echo "Example: $0 tests/test_add_memory_fastapi.py::test_v1_add_memory_1"
    exit 1
fi

TEST_PATH="$1"

echo "🧪 Running single test: $TEST_PATH"
echo "==============================================="

# Run the test in the test-runner container
docker compose run --rm test-runner poetry run pytest "$TEST_PATH" -v -s --tb=short

echo "==============================================="
echo "✅ Test execution complete"
