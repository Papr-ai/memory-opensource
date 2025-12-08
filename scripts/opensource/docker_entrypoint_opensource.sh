#!/bin/bash
# ============================================
# Papr Memory Open Source - Docker Entrypoint
# ============================================
# This script runs on container startup and:
# 1. Waits for Parse Server to be ready
# 2. Checks if this is first run (no users exist)
# 3. Auto-creates default user + workspace + API key
# 4. Writes API key to /app/.env.generated for user to copy
# 5. Starts the main application

set -e

echo "🚀 Papr Memory Open Source - Starting..."
echo "=========================================="

# Wait for Neo4j to be ready
echo "⏳ Waiting for Neo4j to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if cypher-shell -u "${NEO4J_USERNAME:-neo4j}" -p "${NEO4J_PASSWORD:-password}" \
        -a "${NEO4J_URL:-bolt://neo4j:7687}" \
        "RETURN 1" > /dev/null 2>&1; then
        echo "✅ Neo4j is ready!"
        break
    fi

    attempt=$((attempt + 1))
    if [ $attempt -eq $max_attempts ]; then
        echo "⚠️  Neo4j not ready after ${max_attempts} attempts - continuing anyway"
        break
    fi

    echo "   Attempt $attempt/$max_attempts - waiting 2s..."
    sleep 2
done

# Wait for Qdrant to be ready
echo "⏳ Waiting for Qdrant to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    # Check if Qdrant API is responding (try /collections endpoint)
    if curl -s -f "${QDRANT_URL:-http://qdrant:6333}/collections" > /dev/null 2>&1; then
        echo "✅ Qdrant is ready!"
        break
    fi

    attempt=$((attempt + 1))
    if [ $attempt -eq $max_attempts ]; then
        echo "⚠️  Qdrant not ready after ${max_attempts} attempts - continuing anyway"
        break
    fi

    echo "   Attempt $attempt/$max_attempts - waiting 2s..."
    sleep 2
done

# Initialize Qdrant collections
echo ""
echo "🔧 Initializing Qdrant collections..."
cd /app
python3 scripts/opensource/init_qdrant_collections_opensource.py || {
    echo "⚠️  Qdrant collection initialization failed - continuing anyway"
}

# Wait for Parse Server to be ready
echo ""
echo "⏳ Waiting for Parse Server to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s -o /dev/null -w "%{http_code}" \
        -H "X-Parse-Application-Id: ${PARSE_APPLICATION_ID}" \
        "${PARSE_SERVER_URL}/health" | grep -q "200"; then
        echo "✅ Parse Server is ready!"
        break
    fi

    attempt=$((attempt + 1))
    if [ $attempt -eq $max_attempts ]; then
        echo "❌ Parse Server failed to start after ${max_attempts} attempts"
        exit 1
    fi

    echo "   Attempt $attempt/$max_attempts - waiting 2s..."
    sleep 2
done

# Check if this is first run (no users exist)
echo ""
echo "🔍 Checking if Parse schemas are initialized..."

# Check if Organization class exists (custom class that we create)
schema_check=$(curl -s \
    -H "X-Parse-Application-Id: ${PARSE_APPLICATION_ID}" \
    -H "X-Parse-Master-Key: ${PARSE_MASTER_KEY}" \
    "${PARSE_SERVER_URL}/schemas/Organization" 2>/dev/null | \
    python3 -c "import sys, json; data = json.load(sys.stdin); print('exists' if 'className' in data else 'missing')" 2>/dev/null || echo "missing")

if [ "$schema_check" = "missing" ]; then
    echo "📋 Schemas not initialized - running schema initialization..."
    echo ""

    cd /app
    python3 scripts/opensource/init_parse_schema_opensource.py \
        --parse-url "${PARSE_SERVER_URL}" \
        --app-id "${PARSE_APPLICATION_ID}" \
        --master-key "${PARSE_MASTER_KEY}" || {
        echo "⚠️  Schema initialization failed - continuing anyway"
    }

    echo ""
else
    echo "✅ Parse schemas already initialized"
fi

echo ""
echo "🔍 Checking if this is first run..."

user_count=$(curl -s \
    -H "X-Parse-Application-Id: ${PARSE_APPLICATION_ID}" \
    -H "X-Parse-Master-Key: ${PARSE_MASTER_KEY}" \
    "${PARSE_SERVER_URL}/classes/_User?limit=0&count=1" | \
    python3 -c "import sys, json; print(json.load(sys.stdin).get('count', 0))" 2>/dev/null || echo "0")

if [ "$user_count" = "0" ]; then
    echo "✨ First run detected! Auto-creating default user..."
    echo ""

    # Generate API key
    API_KEY="pmem_oss_$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"

    # Run bootstrap script
    echo "📝 Creating user: opensource@papr.ai"
    echo "🏢 Creating organization: Papr Open Source"
    echo "🔑 Generating API key..."
    echo ""

    cd /app
    python3 scripts/opensource/bootstrap_opensource_user.py \
        --email "opensource@papr.ai" \
        --name "Open Source User" \
        --organization "Papr Open Source" \
        --api-key "$API_KEY" \
        --parse-url "${PARSE_SERVER_URL}" \
        --app-id "${PARSE_APPLICATION_ID}" \
        --master-key "${PARSE_MASTER_KEY}" || {
        echo "❌ Failed to create default user"
        echo "⚠️  Starting anyway - you can run bootstrap manually"
    }

    # Write API key to generated env file
    cat > /app/.env.generated << EOF
# ==========================================================
# 🎉 Papr Memory Open Source - Auto-Generated Configuration
# ==========================================================
# This file was auto-generated on first run.
# Your API key and credentials are below.
#
# IMPORTANT: Add this to your .env file or export it:
#
#   export PAPR_API_KEY=${API_KEY}
#
# ==========================================================

# Your API Key (use this in API requests)
PAPR_API_KEY=${API_KEY}

# Default User Credentials (for Parse Dashboard)
# URL: http://localhost:4040
# Username: opensource@papr.ai
# Password: Check bootstrap output above
# ==========================================================

# Test your API:
# curl -X POST http://localhost:5001/v1/memory \\
#   -H "X-API-Key: ${API_KEY}" \\
#   -H "Content-Type: application/json" \\
#   -d '{"content": "Hello Papr Memory!", "type": "text"}'

EOF

    echo ""
    echo "=========================================="
    echo "✅ Auto-Bootstrap Complete!"
    echo "=========================================="
    echo ""
    echo "📋 Your API Key:"
    echo "   ${API_KEY}"
    echo ""
    echo "📧 Default Login (Parse Dashboard):"
    echo "   Email: opensource@papr.ai"
    echo "   Password: (check output above)"
    echo ""
    echo "📝 Configuration saved to: /app/.env.generated"
    echo "   Copy this to your host with:"
    echo "   docker cp papr-memory:/app/.env.generated ./"
    echo ""
    echo "🧪 Test your API:"
    echo "   curl -X POST http://localhost:5001/v1/memory \\"
    echo "     -H \"X-API-Key: ${API_KEY}\" \\"
    echo "     -H \"Content-Type: application/json\" \\"
    echo "     -d '{\"content\": \"Hello Papr Memory!\", \"type\": \"text\"}'"
    echo ""
    echo "=========================================="
    echo ""
else
    echo "✅ Existing users found ($user_count) - skipping bootstrap"
    echo ""
fi

# Start the main application
echo "🎯 Starting Papr Memory API..."
echo "=========================================="
echo ""

exec "$@"
