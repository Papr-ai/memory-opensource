#!/bin/bash
# Stop all PAPR Memory services (web + workers) from both compose files

set -e

echo "🛑 Stopping all PAPR Memory services..."
echo "========================================"
echo ""

# Stop docker-compose.yaml services
echo "1️⃣  Stopping services from docker-compose.yaml..."
docker-compose -f docker-compose.yaml down --remove-orphans 2>/dev/null || echo "   No services from docker-compose.yaml"

# Stop docker-compose-split.yaml services
echo "2️⃣  Stopping services from docker-compose-split.yaml..."
docker-compose -f docker-compose-split.yaml down --remove-orphans 2>/dev/null || echo "   No services from docker-compose-split.yaml"

# Force stop any remaining memory containers
echo "3️⃣  Stopping any remaining memory containers..."
docker ps -a | grep memory | awk '{print $1}' | xargs -r docker stop 2>/dev/null || echo "   No remaining containers"
docker ps -a | grep memory | awk '{print $1}' | xargs -r docker rm 2>/dev/null || echo "   No remaining containers"

# Remove networks
echo "4️⃣  Removing networks..."
docker network rm memory_default memory_network 2>/dev/null || echo "   Networks already removed"

echo ""
echo "✅ All services stopped!"
echo ""

# Show status
echo "📊 Current Docker containers:"
docker ps -a | grep memory || echo "   No memory containers running"

echo ""
echo "🔧 To start services:"
echo "   Development (all-in-one): docker-compose up -d"
echo "   Production (split):       docker-compose -f docker-compose-split.yaml up -d"
echo ""

