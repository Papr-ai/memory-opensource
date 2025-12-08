# Docker Deployment Guide

This document explains how the PAPR Memory Server runs in Docker with all required services.

## Services Overview

When you run the Docker container, it starts **three services**:

1. **Web Server** (FastAPI) - Port 5001
   - Handles all API requests
   - Defined in `main.py`

2. **Memory Processing Worker** (Temporal)
   - Processes batch memory operations
   - Handles memory indexing, graph generation, metrics
   - Defined in `start_temporal_worker.py`

3. **Document Processing Worker** (Temporal)
   - Processes document uploads
   - Handles PDF parsing, image extraction, LLM memory generation
   - Defined in `start_document_worker.py`

## How It Works

### Startup Script

The container runs `start_all_services.py`, which:
- Starts all three services as separate processes
- Monitors their health
- Handles graceful shutdown
- Restarts failed services

### Process Management

All three services run concurrently in the same container:
- Each service runs in its own Python process
- Logs from all services are combined in Docker logs
- If any service crashes, the supervisor can handle it

## Local Development

### Running Locally (Without Docker)

For development, you can run services separately:

```bash
# Terminal 1: Web server
python main.py

# Terminal 2: Memory worker
python start_temporal_worker.py

# Terminal 3: Document worker
python start_document_worker.py
```

Or run all at once:

```bash
# Python version (recommended)
python start_all_services.py

# Or bash version
./start_all_services.sh
```

### Running with Docker

```bash
# Build and run
docker-compose up --build

# Or with specific image tag
IMAGE_NAME=memory IMAGE_TAG=latest docker-compose up --build
```

### Debugging with Docker

Uncomment the debug command in `docker-compose.yaml`:

```yaml
command: python -m debugpy --wait-for-client --listen 0.0.0.0:5678 start_all_services.py
```

Then connect your debugger to port 5678.

## Production Deployment (Azure)

### Building the Image

```bash
# Set your Azure Container Registry
ACR_NAME="testpaprcontainer.azurecr.io"
IMAGE_NAME="memory"
IMAGE_TAG="v1.0.0"

# Build
docker build -t ${ACR_NAME}/${IMAGE_NAME}:${IMAGE_TAG} .

# Push
docker push ${ACR_NAME}/${IMAGE_NAME}:${IMAGE_TAG}
```

### Environment Variables

Make sure your `.env` file or Azure App Service Configuration includes:

```bash
# Temporal Configuration
TEMPORAL_HOST=us-west-2.aws.api.temporal.io:7233
TEMPORAL_NAMESPACE=papr-memory.pq3ak
TEMPORAL_CLIENT_CERT=<your-cert>
TEMPORAL_CLIENT_KEY=<your-key>

# Parse Configuration
PARSE_SERVER_URL=https://your-parse-server.com
PARSE_APPLICATION_ID=your-app-id
PARSE_MASTER_KEY=your-master-key

# Other services...
```

### Monitoring

View logs for all services:

```bash
# Docker
docker-compose logs -f

# Azure Container Instance
az container logs --name <container-name> --resource-group <resource-group>
```

Logs will show entries from all three services with their respective loggers.

## Troubleshooting

### Worker Not Processing Jobs

1. Check Temporal connection:
   ```bash
   # View logs
   docker-compose logs | grep "Temporal"
   
   # Should see:
   # "Successfully connected to Temporal"
   # "Starting Temporal worker on task queue: memory-processing"
   # "Starting Document Temporal worker on task queue: document-processing"
   ```

2. Check Temporal Cloud dashboard for worker status

### Service Crashes

The supervisor will log which service crashed:

```bash
docker-compose logs | grep "exited unexpectedly"
```

### Memory Issues

If running all services in one container causes memory issues, you can split them:

1. Create separate Dockerfiles for each service
2. Deploy as separate containers/services
3. Or use Kubernetes with separate pods

## Architecture Notes

### Why One Container?

- **Simplicity**: Single deployment unit
- **Shared Resources**: All services share the same Python environment
- **Cost**: One Azure Container Instance instead of three

### When to Split?

Consider separate containers if:
- Individual services need different resource limits
- You want independent scaling (more document workers than memory workers)
- Memory pressure requires isolation
- You need zero-downtime deployments per service

## Files

- `Dockerfile` - Container definition
- `docker-compose.yaml` - Local development orchestration
- `start_all_services.py` - Python supervisor (used in production)
- `start_all_services.sh` - Bash alternative
- `main.py` - Web server
- `start_temporal_worker.py` - Memory processing worker
- `start_document_worker.py` - Document processing worker

