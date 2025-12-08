#!/bin/bash
# Script to check Temporal worker logs and status

set -e

echo "=========================================="
echo "   PAPR Memory - Worker Status Check"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Determine which compose file to use
if [ -f "docker-compose-split.yaml" ] && docker-compose -f docker-compose-split.yaml ps | grep -q "memory-memory-worker"; then
    COMPOSE_FILE="docker-compose-split.yaml"
    echo "📦 Using: docker-compose-split.yaml (split services)"
else
    COMPOSE_FILE="docker-compose.yaml"
    echo "📦 Using: docker-compose.yaml (all-in-one)"
fi

echo ""
echo "1️⃣  Container Status:"
echo "─────────────────────────────────────────"
docker-compose -f $COMPOSE_FILE ps
echo ""

echo "2️⃣  Memory Worker Logs (last 50 lines):"
echo "─────────────────────────────────────────"
if docker-compose -f $COMPOSE_FILE ps | grep -q "memory-worker"; then
    docker-compose -f $COMPOSE_FILE logs --tail=50 memory-worker
else
    echo "⚠️  Memory worker not found (might be in all-in-one mode)"
fi
echo ""

echo "3️⃣  Document Worker Logs (last 50 lines):"
echo "─────────────────────────────────────────"
if docker-compose -f $COMPOSE_FILE ps | grep -q "document-worker"; then
    docker-compose -f $COMPOSE_FILE logs --tail=50 document-worker
else
    echo "⚠️  Document worker not found (might be in all-in-one mode)"
fi
echo ""

echo "4️⃣  Web Server Health:"
echo "─────────────────────────────────────────"
if curl -s http://localhost:5001/health > /dev/null 2>&1; then
    echo "✅ Web server is healthy"
    curl -s http://localhost:5001/health | jq .
else
    echo "❌ Web server not responding"
fi
echo ""

echo "5️⃣  Resource Usage:"
echo "─────────────────────────────────────────"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
echo ""

echo "=========================================="
echo "   Follow Live Logs:"
echo "=========================================="
echo ""
echo "All services:     docker-compose -f $COMPOSE_FILE logs -f"
echo "Memory worker:    docker-compose -f $COMPOSE_FILE logs -f memory-worker"
echo "Document worker:  docker-compose -f $COMPOSE_FILE logs -f document-worker"
echo "Web server:       docker-compose -f $COMPOSE_FILE logs -f web"
echo ""

