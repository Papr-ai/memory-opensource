# Embedding Model Configuration Guide

## Overview

Papr Memory supports multiple embedding models for semantic search. This guide explains how to choose and configure the right model for your needs.

## Quick Start

### For New Users (Recommended)
Use the default **Qwen3-Embedding-0.6B** model - it's fast, efficient, and works great for most use cases:

```bash
# In your .env file:
USE_LOCAL_EMBEDDINGS=true
LOCAL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
LOCAL_EMBEDDING_DIMENSIONS=1024
```

Then start services:
```bash
docker compose up -d
```

### For Existing Users with Qwen4B

If you're already using the 4B model and want to keep it:

```bash
# In your .env file:
USE_LOCAL_EMBEDDINGS=true
LOCAL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
LOCAL_EMBEDDING_DIMENSIONS=2560
```

**Important**: Make sure your Qdrant collection matches the dimensions. If you get dimension errors, see [Troubleshooting](#troubleshooting) below.

## Available Models

### Option 1: Qwen3-Embedding-0.6B (Recommended)
- **Size**: ~1.2GB download
- **Dimensions**: 1024
- **Context**: 32k tokens
- **Speed**: Fast ⚡
- **Best for**: Most users, production deployments, limited hardware

### Option 2: Qwen3-Embedding-4B
- **Size**: ~8GB download
- **Dimensions**: 2560
- **Context**: 32k tokens
- **Speed**: Slower 🐢
- **Best for**: Highest quality embeddings, research, powerful hardware

## How It Works

The system **automatically** selects the correct Qdrant collection based on your `LOCAL_EMBEDDING_DIMENSIONS`:

- **1024 dimensions** → Uses `Qwen0pt6B` collection
- **2560 dimensions** → Uses `Qwen4B` collection

You don't need to manually configure collection names - the system handles it!

## Switching Models

### From 0.6B to 4B (or vice versa)

**Option 1: Clean Start (Recommended)**
```bash
# Stop and remove all data
docker compose down -v

# Update your .env file
# Change LOCAL_EMBEDDING_MODEL and LOCAL_EMBEDDING_DIMENSIONS

# Start fresh
docker compose up -d
```

**Option 2: Keep Existing Data**
```bash
# Delete only the Qdrant collection
curl -X DELETE http://localhost:6333/collections/Qwen0pt6B
# or
curl -X DELETE http://localhost:6333/collections/Qwen4B

# Update your .env file
# Change LOCAL_EMBEDDING_MODEL and LOCAL_EMBEDDING_DIMENSIONS

# Restart services
docker compose restart papr-memory temporal-worker
```

**Note**: With Option 2, you'll need to re-create all your memories as the embeddings will be regenerated.

## Troubleshooting

### Error: "Vector dimension error: expected dim: 2560, got 1024"

This means:
- Your `.env` says to use **0.6B model (1024 dims)**
- But Qdrant has a collection configured for **4B model (2560 dims)**

**Fix:**
```bash
# Option 1: Switch to 4B model (to match existing collection)
# In .env:
LOCAL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
LOCAL_EMBEDDING_DIMENSIONS=2560

# Option 2: Delete collection and use 0.6B model
curl -X DELETE http://localhost:6333/collections/Qwen4B
docker compose restart papr-memory temporal-worker
```

### Error: "Vector dimension error: expected dim: 1024, got 2560"

This means:
- Your `.env` says to use **4B model (2560 dims)**
- But Qdrant has a collection configured for **0.6B model (1024 dims)**

**Fix:**
```bash
# Option 1: Switch to 0.6B model (to match existing collection)
# In .env:
LOCAL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
LOCAL_EMBEDDING_DIMENSIONS=1024

# Option 2: Delete collection and use 4B model
curl -X DELETE http://localhost:6333/collections/Qwen0pt6B
docker compose restart papr-memory temporal-worker
```

## Performance Comparison

| Model | Embedding Time* | Memory Usage | Quality |
|-------|----------------|--------------|---------|
| 0.6B  | ~100ms        | ~2GB RAM     | Good ✓  |
| 4B    | ~400ms        | ~10GB RAM    | Better ✓✓ |

*Per batch of 10 memories on Apple M1

## Cloud Embeddings (Alternative)

If you prefer using cloud APIs instead of local models:

```bash
# In .env:
USE_LOCAL_EMBEDDINGS=false
DEEPINFRA_TOKEN=your-deepinfra-token
```

This uses DeepInfra's hosted embedding API (requires API key).

## Docker Resource Requirements

### For 0.6B Model
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: 2 cores minimum, 4 cores recommended
- **Disk**: 10GB for model + data

### For 4B Model
- **RAM**: 12GB minimum, 16GB recommended
- **CPU**: 4 cores minimum, 8 cores recommended
- **Disk**: 20GB for model + data

Set in Docker Desktop → Settings → Resources.

## FAQ

**Q: Can I use both models at the same time?**
A: No, you must choose one model. The system uses a single embedding model for all memories.

**Q: What happens to my existing memories if I switch models?**
A: You'll need to re-create them. Different models produce different embeddings (different dimensions), so they're not compatible.

**Q: Which model should I use?**
A: For most users, **0.6B is recommended**. It's faster, uses less memory, and provides good quality. Only use 4B if you need the highest possible quality and have powerful hardware.

**Q: How do I check which model I'm currently using?**
A: Check your `.env` file for `LOCAL_EMBEDDING_MODEL` and `LOCAL_EMBEDDING_DIMENSIONS`.

**Q: Can I use OpenAI embeddings instead?**
A: Not currently supported in open source. Local Qwen models or DeepInfra are the supported options.

## Related Documentation

- [Testing Guide](TESTING.md)
- [Docker Setup](../README.md#-docker-setup)
- [Configuration](../README.md#-configuration)
