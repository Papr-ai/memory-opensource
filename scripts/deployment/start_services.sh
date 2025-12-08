#!/bin/bash
# Start PAPR Memory services with proper cleanup and validation

set -e

COMPOSE_FILE="${1:-docker-compose-split.yaml}"

echo "🚀 Starting PAPR Memory Services"
echo "================================="
echo "Using: $COMPOSE_FILE"
echo ""

# Step 1: Clean up any existing containers
echo "1️⃣  Cleaning up existing containers..."
docker-compose -f docker-compose.yaml down --remove-orphans 2>/dev/null || true
docker-compose -f docker-compose-split.yaml down --remove-orphans 2>/dev/null || true
echo "   ✅ Cleanup complete"
echo ""

# Step 2: Validate compose file
echo "2️⃣  Validating compose file..."
if docker-compose -f $COMPOSE_FILE config -q; then
    echo "   ✅ Compose file is valid"
else
    echo "   ❌ Compose file has errors!"
    exit 1
fi
echo ""

# Step 3: Check .env file
echo "3️⃣  Checking .env file..."
if [ -f ".env" ]; then
    echo "   ✅ .env file exists"
    
    # Check for critical variables
    REQUIRED_VARS=("TEMPORAL_CLOUD_NAMESPACE" "MONGODB_URI" "NEO4J_URL")
    for VAR in "${REQUIRED_VARS[@]}"; do
        if grep -q "^${VAR}=" .env; then
            echo "   ✅ $VAR is set"
        else
            echo "   ⚠️  $VAR not found in .env"
        fi
    done
else
    echo "   ❌ .env file not found!"
    exit 1
fi
echo ""

# Step 4: Check Docker resources
echo "4️⃣  Checking Docker resources..."
DOCKER_MEM=$(docker info --format '{{.MemTotal}}' 2>/dev/null | awk '{print int($1/1024/1024/1024)}')
if [ ! -z "$DOCKER_MEM" ]; then
    echo "   Docker Memory: ${DOCKER_MEM}GB"
    if [ "$DOCKER_MEM" -lt 6 ]; then
        echo "   ⚠️  Warning: Docker has less than 6GB RAM"
        echo "   ⚠️  Increase Docker Desktop memory to 8GB for best performance"
    else
        echo "   ✅ Docker has sufficient memory"
    fi
else
    echo "   ⚠️  Could not determine Docker memory"
fi
echo ""

# Step 5: Build images if needed
echo "5️⃣  Building Docker images..."
docker-compose -f $COMPOSE_FILE build --quiet
echo "   ✅ Images built"
echo ""

# Step 6: Start services
echo "6️⃣  Starting services..."
if [ "$2" == "--logs" ]; then
    # Start with logs (foreground)
    echo "   Starting in foreground (press Ctrl+C to stop)..."
    echo ""
    docker-compose -f $COMPOSE_FILE up
else
    # Start in background
    docker-compose -f $COMPOSE_FILE up -d
    echo "   ✅ Services started in background"
    echo ""
    
    # Step 7: Wait for services to be ready
    echo "7️⃣  Waiting for services to be ready..."
    echo "   This may take 30-60 seconds..."
    
    # Wait for web server
    MAX_WAIT=60
    COUNTER=0
    while [ $COUNTER -lt $MAX_WAIT ]; do
        if docker-compose -f $COMPOSE_FILE ps | grep -q "web.*Up"; then
            echo "   ✅ Web server is up"
            break
        fi
        sleep 2
        COUNTER=$((COUNTER+2))
        echo -n "."
    done
    echo ""
    
    # Check health
    sleep 5
    if curl -s http://localhost:5001/health > /dev/null 2>&1; then
        echo "   ✅ Web server is healthy"
    else
        echo "   ⚠️  Web server not responding yet (may still be starting)"
    fi
    echo ""
    
    # Step 8: Show status
    echo "8️⃣  Service Status:"
    docker-compose -f $COMPOSE_FILE ps
    echo ""
    
    echo "✅ Services started successfully!"
    echo ""
    echo "📊 Useful commands:"
    echo "   View logs:        docker-compose -f $COMPOSE_FILE logs -f"
    echo "   View web logs:    docker-compose -f $COMPOSE_FILE logs -f web"
    echo "   View worker logs: docker-compose -f $COMPOSE_FILE logs -f workers"
    echo "   Check status:     docker-compose -f $COMPOSE_FILE ps"
    echo "   Stop services:    docker-compose -f $COMPOSE_FILE down"
    echo "   Check health:     curl http://localhost:5001/health"
    echo ""
    echo "🌐 Web server: http://localhost:5001"
    echo "📖 API docs:   http://localhost:5001/docs"
    echo ""
fi

