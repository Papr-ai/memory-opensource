#!/bin/bash
# Test Neo4j GraphQL endpoint with JWT token from your deployed server

echo "🔍 Testing Neo4j GraphQL with JWT authentication"
echo "================================================"
echo ""

# First, get a JWT token from your deployed server
echo "Step 1: Getting JWT token from memory server..."
echo ""

# You need your API key from .env
API_KEY=$(grep "^PAPR_MEMORY_API_KEY=" .env 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d "'")

if [ -z "$API_KEY" ]; then
    echo "❌ PAPR_MEMORY_API_KEY not found in .env"
    echo "Please add your API key to .env file"
    exit 1
fi

# Make a request to your server to trigger JWT generation and capture it from logs
# For now, let's test if the endpoint is accessible
echo "Step 2: Testing Neo4j GraphQL endpoint..."
echo ""

# Simple introspection query
QUERY='{"query":"{ __schema { queryType { name } } }"}'

echo "📤 Sending introspection query to Neo4j GraphQL..."
echo "Endpoint: https://de7df98e-graphql.production-orch-0042.neo4j.io/graphql"
echo ""

# This will fail without JWT, but shows us the error
curl -v -X POST "https://de7df98e-graphql.production-orch-0042.neo4j.io/graphql" \
  -H "Content-Type: application/json" \
  -d "$QUERY" \
  2>&1 | grep -E "(< HTTP|Forbidden|error|data)"

echo ""
echo "================================================"
echo "To test with actual JWT:"
echo "1. Make a request through your FastAPI server"
echo "2. Check Azure logs for 'Generated JWT for user'"
echo "3. Copy the JWT token from logs"
echo "4. Use: curl -H 'Authorization: Bearer <token>' ..."
