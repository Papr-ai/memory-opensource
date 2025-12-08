#!/bin/bash
# Diagnose why web container is failing to start

set -e

echo "🔍 Diagnosing Web Container Startup Issues"
echo "=========================================="
echo ""

# Check if any memory containers exist
echo "1️⃣  Current container status:"
docker ps -a | grep memory || echo "   No memory containers found"
echo ""

# Try to see logs from failed web container
echo "2️⃣  Checking web container logs..."
WEB_CONTAINER=$(docker ps -a --filter "name=memory-web" --format "{{.ID}}" | head -1)
if [ ! -z "$WEB_CONTAINER" ]; then
    echo "   Found web container: $WEB_CONTAINER"
    echo "   Last 50 lines of logs:"
    echo "   ─────────────────────────────────────"
    docker logs --tail=50 $WEB_CONTAINER 2>&1
else
    echo "   No web container found"
fi
echo ""

# Check Docker resources
echo "3️⃣  Docker resource usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null || echo "   No containers running"
echo ""

# Check for port conflicts
echo "4️⃣  Checking port 5001 (web server):"
lsof -i :5001 2>/dev/null || echo "   Port 5001 is free"
echo ""

# Check Docker system info
echo "5️⃣  Docker system info:"
docker info --format "Memory: {{.MemTotal}}\nCPUs: {{.NCPU}}" 2>/dev/null
echo ""

# Check Docker Compose files
echo "6️⃣  Validating compose files:"
echo "   docker-compose.yaml:"
docker-compose -f docker-compose.yaml config -q && echo "      ✅ Valid" || echo "      ❌ Invalid"
echo "   docker-compose-split.yaml:"
docker-compose -f docker-compose-split.yaml config -q && echo "      ✅ Valid" || echo "      ❌ Invalid"
echo ""

echo "7️⃣  Recommended actions:"
echo "   - Clean up: docker-compose down --remove-orphans"
echo "   - Try starting with logs: docker-compose up (without -d flag)"
echo "   - Check .env file exists and has all required variables"
echo ""

