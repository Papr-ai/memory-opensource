#!/bin/bash

# Security Schema End-to-End Test Runner
# This script sources the .env file and runs the security monitoring schema test
# with proper configuration for ollama certificate, localhost:8000, and parse-dev

set -e  # Exit on any error

echo "🛡️ Security Schema Test Runner"
echo "================================"

# Check if .env file exists
if [ -f ".env" ]; then
    echo "📋 Sourcing .env file..."
    source .env
    echo "✅ Environment variables loaded"
else
    echo "⚠️ No .env file found - using default environment"
fi

# Set required environment variables for the test
export MEMORY_SERVER_URL="http://localhost:8000"
export PARSE_SERVER_URL="parse-dev"
export OLLAMA_CERTIFICATE_PATH="${OLLAMA_CERTIFICATE_PATH:-/path/to/ollama/cert}"

echo ""
echo "🔧 Test Configuration:"
echo "   Memory Server: ${MEMORY_SERVER_URL}"
echo "   Parse Server: ${PARSE_SERVER_URL}"
echo "   Ollama Cert: ${OLLAMA_CERTIFICATE_PATH}"
echo ""

# Check if memory server is running
echo "🔍 Checking memory server availability..."
if curl -s -f "${MEMORY_SERVER_URL}/health" > /dev/null 2>&1; then
    echo "✅ Memory server is running at ${MEMORY_SERVER_URL}"
else
    echo "❌ Memory server is not accessible at ${MEMORY_SERVER_URL}"
    echo "   Please ensure the memory server is running on localhost:8000"
    exit 1
fi

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest is not installed"
    echo "   Please install pytest: pip install pytest"
    exit 1
fi

echo ""
echo "🚀 Starting Security Schema End-to-End Test..."
echo "================================================"

# Run the specific security test with verbose output
pytest tests/test_security_schema_end_to_end.py \
    -v \
    -s \
    --tb=short \
    --capture=no \
    --log-cli-level=INFO \
    --log-cli-format="%(asctime)s [%(levelname)8s] %(message)s" \
    --log-cli-date-format="%Y-%m-%d %H:%M:%S"

echo ""
echo "🎉 Security Schema Test Completed!"
echo "=================================="



