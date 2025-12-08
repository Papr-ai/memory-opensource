#!/bin/bash
# Start all services for PAPR Memory Server
# This script starts the web server and both Temporal workers

set -e

echo "Starting PAPR Memory Server with all services..."

# Start the web server in the background
echo "Starting web server..."
python main.py &
WEB_PID=$!
echo "Web server started (PID: $WEB_PID)"

# Give web server a moment to initialize
sleep 2

# Start the memory processing Temporal worker in the background
echo "Starting memory processing Temporal worker..."
python start_temporal_worker.py &
MEMORY_WORKER_PID=$!
echo "Memory worker started (PID: $MEMORY_WORKER_PID)"

# Start the document processing Temporal worker in the background
echo "Starting document processing Temporal worker..."
python start_document_worker.py &
DOC_WORKER_PID=$!
echo "Document worker started (PID: $DOC_WORKER_PID)"

echo ""
echo "All services started successfully!"
echo "  - Web server: PID $WEB_PID"
echo "  - Memory worker: PID $MEMORY_WORKER_PID"
echo "  - Document worker: PID $DOC_WORKER_PID"
echo ""

# Function to handle shutdown
cleanup() {
    echo ""
    echo "Shutting down all services..."
    kill $WEB_PID $MEMORY_WORKER_PID $DOC_WORKER_PID 2>/dev/null || true
    wait $WEB_PID $MEMORY_WORKER_PID $DOC_WORKER_PID 2>/dev/null || true
    echo "All services stopped"
    exit 0
}

# Trap SIGTERM and SIGINT to cleanup
trap cleanup SIGTERM SIGINT

# Wait for any process to exit
wait -n

# If any process exits, cleanup and exit
cleanup

