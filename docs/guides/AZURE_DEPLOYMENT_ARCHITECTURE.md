# Azure Deployment Architecture for PAPR Memory

## Overview
This document outlines the recommended Azure architecture for deploying PAPR Memory with Temporal workers.

## Architecture Options

### Option 1: Single App Service (Development/Small Scale)
**Cost**: ~$50-100/month
```
┌─────────────────────────────────────────┐
│   Azure App Service (P1v3)              │
│   - Web Server (FastAPI)                │
│   - Memory Worker (Temporal)            │
│   - Document Worker (Temporal)          │
│   Memory: 8GB RAM, 2 vCPU               │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│   Self-Hosted Temporal Server           │
│   (Azure Container Instance)            │
│   Memory: 4GB RAM, 2 vCPU               │
│   Cost: ~$40/month                      │
└─────────────────────────────────────────┘
```

**When to use**: 
- < 1000 documents/month
- < 10,000 memory operations/month
- Development/staging

**Setup**:
```yaml
# Use docker-compose.yaml (all-in-one)
deploy:
  resources:
    limits:
      memory: 8G
```

---

### Option 2: Split App Services (Production - RECOMMENDED)
**Cost**: ~$150-250/month
```
┌──────────────────────────────┐
│  App Service 1: Web Server   │
│  (P1v3: 8GB RAM, 2 vCPU)     │
│  - FastAPI endpoints         │
│  - Handles user requests     │
│  Cost: ~$100/month           │
└──────────────────────────────┘

┌──────────────────────────────┐
│  App Service 2: Workers      │
│  (P2v3: 16GB RAM, 4 vCPU)    │
│  - Memory Worker             │
│  - Document Worker           │
│  Cost: ~$200/month           │
└──────────────────────────────┘
         ↓
┌──────────────────────────────┐
│  Self-Hosted Temporal        │
│  (Container Instance)        │
│  Cost: ~$40/month            │
└──────────────────────────────┘
```

**When to use**:
- > 1000 documents/month
- > 10,000 memory operations/month
- Production workloads
- Need independent scaling

**Benefits**:
✅ API stays responsive during heavy document processing
✅ Scale workers independently
✅ Better monitoring and debugging
✅ Fault isolation

**Setup**:
```yaml
# Use docker-compose-split.yaml
# Deploy to two separate App Services
```

---

### Option 3: Kubernetes (Enterprise Scale)
**Cost**: ~$500+/month
```
┌────────────────────────────────────────┐
│   Azure Kubernetes Service (AKS)       │
│   - Web Server (3 replicas)            │
│   - Memory Worker (2 replicas)         │
│   - Document Worker (3 replicas)       │
│   - Temporal (StatefulSet)             │
│   - Auto-scaling enabled                │
└────────────────────────────────────────┘
```

**When to use**:
- > 10,000 documents/month
- > 100,000 memory operations/month
- Need auto-scaling
- Multiple environments (dev/staging/prod)

---

## Temporal Hosting Options

### Self-Hosted (Recommended)
```bash
# Deploy Temporal to Azure Container Instance
az container create \
  --resource-group papr-memory \
  --name temporal-server \
  --image temporalio/auto-setup:latest \
  --cpu 2 \
  --memory 4 \
  --ports 7233 \
  --environment-variables \
    DB=postgresql \
    DB_PORT=5432 \
    POSTGRES_USER=temporal \
    POSTGRES_PWD=your-password \
    POSTGRES_SEEDS=your-postgres.postgres.database.azure.com
```

**Cost breakdown**:
- Container Instance: ~$40/month
- PostgreSQL (Basic tier): ~$30/month
- **Total**: ~$70/month

### Temporal Cloud (Optional)
**Pricing**:
- Starter: $200/month (10K actions/month)
- Pro: $500+/month (100K actions/month)

**Only worth it if**:
- You need 99.99% SLA
- Don't want to manage Temporal infrastructure
- Enterprise compliance requirements

---

## Cost Comparison

| Architecture | Monthly Cost | Best For |
|--------------|--------------|----------|
| **Option 1: Single App Service** | $90-140 | Development, < 1K docs/month |
| **Option 2: Split App Services** | $310-370 | Production, 1-10K docs/month |
| **Option 3: AKS** | $500+ | Enterprise, > 10K docs/month |

**Key insight**: You're already paying for Azure infrastructure. Don't pay for Temporal Cloud unless you have enterprise needs!

---

## Memory Requirements

### Single Machine (Option 1)
```yaml
Web Server: 2GB base + 1GB per request spike = 3-4GB
Memory Worker: 1-2GB
Document Worker: 2-3GB (OCR models)
---
Total: 6-9GB → Use 8GB App Service (P1v3)
```

### Split Services (Option 2)
```yaml
App Service 1 (Web):
  - 4GB for FastAPI
  - 2GB for connections (Neo4j, MongoDB, Qdrant)
  Total: 6-8GB → Use 8GB (P1v3)

App Service 2 (Workers):
  - 3GB for Memory Worker
  - 4GB for Document Worker (TensorLake/Gemini models)
  - 2GB buffer
  Total: 9-12GB → Use 16GB (P2v3)
```

---

## Recommended Setup for You

Based on your current stage, I recommend **Option 2: Split App Services**

### Why?
1. ✅ Your API needs to stay responsive (user-facing)
2. ✅ Document processing is heavy (can slow down API)
3. ✅ Independent scaling (more users = scale web, more docs = scale workers)
4. ✅ Better debugging (separate logs)
5. ✅ Reasonable cost (~$310/month vs $700+ for Temporal Cloud)

### Setup Steps

#### 1. Deploy Web Server
```bash
# App Service 1: Web Server
az webapp create \
  --resource-group papr-memory \
  --plan papr-web-plan \
  --name papr-memory-api \
  --deployment-container-image-name testpaprcontainer.azurecr.io/memory:latest

# Configure
az webapp config appsettings set \
  --resource-group papr-memory \
  --name papr-memory-api \
  --settings DOCKER_CUSTOM_IMAGE_NAME=testpaprcontainer.azurecr.io/memory:latest \
             WEBSITES_PORT=5001 \
             STARTUP_COMMAND="poetry run uvicorn main:app --host 0.0.0.0 --port 5001"
```

#### 2. Deploy Workers
```bash
# App Service 2: Workers
az webapp create \
  --resource-group papr-memory \
  --plan papr-workers-plan \
  --name papr-memory-workers \
  --deployment-container-image-name testpaprcontainer.azurecr.io/memory:latest

# Configure
az webapp config appsettings set \
  --resource-group papr-memory \
  --name papr-memory-workers \
  --settings DOCKER_CUSTOM_IMAGE_NAME=testpaprcontainer.azurecr.io/memory:latest \
             STARTUP_COMMAND="python start_all_workers.py"
```

---

## Monitoring

### Web Server Health
```bash
curl https://papr-memory-api.azurewebsites.net/health
```

### Worker Health
```bash
# Check Temporal UI
http://your-temporal-server:8080

# Check worker logs
az webapp log tail --name papr-memory-workers --resource-group papr-memory
```

---

## Next Steps

1. ✅ Start with **Option 1** (single App Service) for development
2. ✅ Monitor resource usage with `docker stats`
3. ✅ When you hit 1K docs/month, migrate to **Option 2** (split services)
4. ✅ Self-host Temporal (save $200-1000/month)
5. ✅ Use Azure Monitor for alerting

---

## Questions?

- **Do I need Temporal Cloud?** No, self-host saves you $200-1000/month
- **One or two App Services?** Two for production, one for dev
- **How much memory?** 8GB for single, 8GB + 16GB for split
- **When to use Kubernetes?** Only when you're processing 10K+ docs/month

