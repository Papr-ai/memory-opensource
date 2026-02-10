#!/bin/bash
# Simple wrapper to run all V1 open source tests in Docker
# Reports are saved to tests/test_reports/
#
# Usage:
#   ./run_tests.sh                    # Run all OSS tests
#   ./run_tests.sh --quick            # Pass args to the test runner

cd "$(dirname "$0")" && ./scripts/run_tests_docker.sh "$@"
