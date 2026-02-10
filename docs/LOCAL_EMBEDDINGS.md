# Local Embeddings Configuration

Papr Memory supports both **local (on-device)** and **cloud-based** embedding generation for your memory items. By default, the open-source edition uses local embeddings to ensure privacy and eliminate dependency on external APIs.

## Overview

### Local Embeddings (Default)
- **Model**: Qwen3-Embedding-0.6B
- **Size**: ~1.2GB download on first use
- **Dimensions**: 1024 (configurable)
- **Context Length**: Up to 32,768 tokens
- **Languages**: 100+ languages including programming languages
- **Pros**: 
  - ✅ Complete privacy - no data leaves your device
  - ✅ No API costs
  - ✅ No rate limits
  - ✅ Works offline
- **Cons**: 
  - ⚠️ Slower than cloud APIs on CPU (faster on GPU)
  - ⚠️ Requires ~1.2GB disk space
  - ⚠️ Uses system memory (~2-3GB RAM)

### Cloud Embeddings (Optional)
- **Model**: Qwen3-Embedding-4B (via DeepInfra or Vertex AI)
- **Dimensions**: 2560
- **Pros**: 
  - ✅ Faster processing (especially for large batches)
  - ✅ Higher dimensional embeddings (better quality)
  - ✅ No local compute required
- **Cons**: 
  - ⚠️ Requires API key and costs money
  - ⚠️ Rate limits apply
  - ⚠️ Data sent to external API

## Configuration

### Using Local Embeddings (Default)

Local embeddings are enabled by default. No additional configuration is required!

```bash
# In your .env file (default configuration)
USE_LOCAL_EMBEDDINGS=true
LOCAL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
LOCAL_EMBEDDING_DIMENSIONS=1024
```

On first use, the model will be automatically downloaded from Hugging Face (~1.2GB). This may take a few minutes depending on your internet connection.

### Switching to Cloud Embeddings

If you prefer to use cloud-based embeddings for faster processing or higher quality:

1. **Update your `.env` file**:

```bash
# Disable local embeddings
USE_LOCAL_EMBEDDINGS=false

# Configure cloud API (choose one)
# Option 1: DeepInfra (recommended)
DEEPINFRA_TOKEN=your_deepinfra_token_here
DEEPINFRA_API_URL=https://api.deepinfra.com/v1/openai/embeddings

# Option 2: Vertex AI (Google Cloud)
USE_VERTEX_AI=true
VERTEX_AI_PROJECT=your_project_id
VERTEX_AI_ENDPOINT_ID=your_endpoint_id
VERTEX_AI_LOCATION=us-west1
```

2. **Restart your services**:

```bash
docker compose restart papr-memory
```

### GPU Acceleration (Optional)

If you have an NVIDIA GPU with CUDA support, local embeddings will automatically use it for faster processing:

```bash
# Check if GPU is available
docker compose exec papr-memory python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

For GPU support in Docker, you'll need:
- NVIDIA Container Toolkit installed
- Add GPU support to your `docker-compose.yaml`:

```yaml
services:
  papr-memory:
    # ... other config ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Switching Between Models

### Migrating from Cloud to Local

If you've been using cloud embeddings and want to switch to local:

1. Update your `.env` to enable local embeddings
2. Clear your Qdrant collections (dimensions are different):

```bash
# Stop services
docker compose down

# Remove Qdrant data
docker volume rm memory-opensource_qdrant_data

# Restart services (collections will be recreated with correct dimensions)
docker compose up -d
```

3. Re-process your memories to generate new embeddings

### Migrating from Local to Cloud

Reverse process:

1. Update your `.env` to disable local embeddings and add API credentials
2. Clear and recreate Qdrant collections (dimensions change from 1024 to 2560)
3. Re-process your memories

## Performance Comparison

### Local Embeddings (CPU)
- Single item: ~0.5-2 seconds
- Batch of 10 items: ~5-15 seconds
- Best for: Small-scale deployments, privacy-focused users

### Local Embeddings (GPU)
- Single item: ~0.1-0.3 seconds
- Batch of 10 items: ~1-3 seconds
- Best for: Medium-scale deployments with GPU access

### Cloud Embeddings (DeepInfra)
- Single item: ~0.2-0.5 seconds
- Batch of 10 items: ~1-2 seconds (parallel processing)
- Best for: High-volume deployments, fastest possible processing

## Troubleshooting

### Model Download Fails

If the model download fails or is slow:

1. **Check your internet connection**
2. **Manually download the model**:

```bash
# Inside the container
docker compose exec papr-memory python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True)"
```

3. **Check Hugging Face status**: https://status.huggingface.co/

### Out of Memory Errors

If you see out-of-memory errors:

1. **Reduce batch size** in `.env`:

```bash
MAX_BATCH_SIZE=10  # Reduce from default 50
```

2. **Increase Docker memory limit** in Docker Desktop settings (Mac/Windows)

3. **Switch to cloud embeddings** if local resources are insufficient

### Slow Performance

If local embeddings are too slow:

1. **Enable GPU acceleration** (see above)
2. **Switch to cloud embeddings** for faster processing
3. **Reduce batch size** to process fewer items at once

## Model Details

### Qwen3-Embedding-0.6B (Local)

- **Source**: Alibaba Cloud
- **Parameters**: 0.6 billion
- **Architecture**: Transformer-based
- **Max Context**: 32,768 tokens
- **Output Dimensions**: 32-1024 (configurable, default 1024)
- **License**: Apache 2.0
- **More Info**: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

### Qwen3-Embedding-4B (Cloud)

- **Source**: Alibaba Cloud (via DeepInfra/Vertex AI)
- **Parameters**: 4 billion
- **Output Dimensions**: 2560
- **Quality**: Higher quality than 0.6B model
- **Speed**: Faster inference on optimized cloud infrastructure

## Best Practices

1. **Start with local embeddings** - Great default for most users
2. **Monitor performance** - Check embedding generation times in logs
3. **Consider GPU** - If processing large volumes locally
4. **Switch to cloud** - If local performance is insufficient
5. **Keep backups** - Before switching models (dimensions change)
6. **Test both** - Compare quality and performance for your use case

## Security & Privacy

### Local Embeddings
- ✅ **All data stays on your device**
- ✅ No data sent to external services
- ✅ Compliant with strict data residency requirements
- ✅ Works in air-gapped environments (after initial model download)

### Cloud Embeddings
- ⚠️ **Data sent to external API** (DeepInfra, Vertex AI, etc.)
- ⚠️ Subject to API provider's terms and privacy policy
- ⚠️ Consider for non-sensitive data only
- ⚠️ Ensure compliance with your organization's data policies

## Support

For issues or questions:
- GitHub Issues: https://github.com/papr-org/memory-opensource/issues
- Documentation: https://docs.papr.so
- Discord: https://discord.gg/papr
