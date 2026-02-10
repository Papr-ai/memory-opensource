# Papr Memory - Open Source Quick Start

Get started with Papr Memory open source in 5 minutes!

## Prerequisites

- Docker & Docker Compose
- API Keys: OpenAI API key, Groq API key, and Deep Infra API key
  - Note: Hugging Face is also supported, and local Qwen on-device support will be added soon

## Step 1: Configure Environment

```bash
# Copy example environment file
cp .env.example .env.opensource

# Edit .env.opensource and add your API keys (OpenAI, Groq, Deep Infra)
# Required: OPENAI_API_KEY, GROQ_API_KEY, DEEPINFRA_API_KEY
```

**Note**: The docker compose file uses `.env.opensource` specifically for open-source setup.

## Step 2: Start Services

```bash
# Start all services (MongoDB, Redis, Neo4j, Qdrant, Parse Server, API)
docker compose -f docker compose.yaml up -d

# Check status
docker compose -f docker compose.yaml ps

# View logs (watch for auto-initialization)
docker compose -f docker compose.yaml logs -f papr-memory
```

**What happens automatically on first run:**
- ✅ Waits for all services (Neo4j, Qdrant, Parse Server) to be ready
- ✅ Initializes Qdrant collections (`neo4j_properties`, `neo4j_properties_dev`, `Qwen4B`)
- ✅ Initializes Parse Server schemas (if missing)
- ✅ Creates default user account (`opensource@papr.ai`)
- ✅ Generates API key automatically
- ✅ **Saves test credentials to `.env.opensource` on your host** (auto-synced via volume mount)
- ✅ Saves additional credentials to `/app/.env.generated` inside container

## Step 3: Get Your API Key

On first run, the container automatically creates a default user and generates an API key. **Test credentials are automatically written to your `.env.opensource` file** - no manual copying needed!

```bash
# View your API key and test credentials
grep "TEST_" .env.opensource

# Or check the container logs for the API key
docker compose logs papr-memory | grep "API Key"
```

**Default credentials:**
- **Email**: `opensource@papr.ai`
- **Password**: Check bootstrap output in logs (auto-generated)
- **API Key**: Auto-generated, shown in logs and saved to `.env.generated`

## Step 4: Test the API

```bash
# Replace YOUR_API_KEY with the key from Step 3
export API_KEY="your-api-key-from-logs"

# Test memory creation
curl -X POST http://localhost:5001/v1/memory \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "This is my first memory!",
    "type": "text"
  }'

# Search memories
curl -X POST http://localhost:5001/v1/memory/search \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "first memory",
    "max_memories": 10
  }'
```

## Available Services

Once running, you have access to:

- **API**: http://localhost:5001
- **API Docs (Swagger)**: http://localhost:5001/docs
- **API Docs (ReDoc)**: http://localhost:5001/redoc
- **Parse Dashboard**: http://localhost:4040 (optional, requires `--profile dashboard`)
- **Neo4j Browser**: http://localhost:7474 (login: neo4j / password)
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **MongoDB**: localhost:27017 (admin / password)

### Optional: Enable Parse Dashboard

```bash
# Start with Parse Dashboard enabled
docker compose --profile dashboard up -d parse-dashboard

# Access at http://localhost:4040
# Login: admin / password
```

**Tip**: Parse Dashboard is useful for:
- Viewing and managing Parse Server data
- Getting session tokens for testing (view user sessions in the `_Session` class)
- Debugging authentication issues
- Exploring your data structure

## API Key Management

### Generate Additional Keys

```bash
# Run inside the container
docker exec -it papr-memory python scripts/generate_api_key.py \
  --email user@example.com \
  --name "Production Key" \
  --rate-limit 5000

# Or run locally (requires Python environment setup)
python scripts/generate_api_key.py \
  --email user@example.com \
  --name "Production Key" \
  --rate-limit 5000
```

### List All Keys

```bash
# Run inside the container
docker exec -it papr-memory python scripts/generate_api_key.py --list

# Or run locally
python scripts/generate_api_key.py --list
```

### Rate Limits

- Default: 1000 requests/hour per API key
- Configurable when creating keys with `--rate-limit` flag
- No storage limits in open source

## Authentication Methods

### Method 1: API Key (Recommended)

```bash
curl -H "X-API-Key: YOUR_API_KEY" http://localhost:5001/v1/memory
```

### Method 2: Session Token (Advanced)

For multi-user applications:

1. Get session token via Parse REST API or Parse Dashboard
2. Use session token in `X-Session-Token` header

**Option A: Via Parse REST API**
```bash
# Login to get session token
curl -X POST http://localhost:1337/parse/login \
  -H "X-Parse-Application-Id: papr-app-id" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "opensource@papr.ai",
    "password": "your-password"
  }'

# Use session token in API calls
curl -H "X-Session-Token: YOUR_SESSION_TOKEN" \
  http://localhost:5001/v1/memory
```

**Option B: Via Parse Dashboard (Easier for Testing)**
1. Start Parse Dashboard: `docker compose --profile dashboard up -d parse-dashboard`
2. Login at http://localhost:4040 (admin / password)
3. Navigate to `_Session` class to view active session tokens
4. Copy the `sessionToken` field from any active session
5. Use it in API calls with `X-Session-Token` header

## Common Operations

### Add Memory

```bash
curl -X POST http://localhost:5001/v1/memory \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Meeting notes: Discussed Q4 roadmap",
    "type": "text",
    "metadata": {
      "source": "meeting",
      "date": "2025-11-24"
    }
  }'
```

### Search Memories

```bash
curl -X POST http://localhost:5001/v1/memory/search \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "roadmap",
    "max_memories": 10,
    "max_nodes": 10
  }'
```

### Get Memory by ID

```bash
curl -X GET http://localhost:5001/v1/memory/{memory_id} \
  -H "X-API-Key: YOUR_API_KEY"
```

### Batch Add Memories

```bash
curl -X POST http://localhost:5001/v1/memory/batch \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "memories": [
      {"content": "First memory", "type": "text"},
      {"content": "Second memory", "type": "text"},
      {"content": "Third memory", "type": "text"}
    ]
  }'
```

## Troubleshooting

### Services Not Starting

```bash
# Check logs for all services
docker compose -f docker compose.yaml logs

# Check specific service logs
docker compose -f docker compose.yaml logs papr-memory
docker compose -f docker compose.yaml logs mongodb
docker compose -f docker compose.yaml logs neo4j

# Restart specific service
docker compose -f docker compose.yaml restart papr-memory
```

### API Key Not Found

```bash
# Check if auto-bootstrap ran successfully
docker compose -f docker compose.yaml logs papr-memory | grep "API Key"

# Manually run bootstrap (inside container)
docker exec -it papr-memory python scripts/bootstrap_opensource_user.py \
  --email "your@email.com" \
  --name "Your Name" \
  --organization "Your Company"

# Or generate API key manually
docker exec -it papr-memory python scripts/generate_api_key.py \
  --email your@email.com \
  --name "My Project"
```

### Parse Schema Not Initialized

The entrypoint script automatically initializes schemas on first run. If you need to re-initialize:

```bash
# Run inside container
docker exec -it papr-memory python scripts/init_parse_schema_opensource.py \
  --parse-url http://parse-server:1337/parse \
  --app-id papr-oss-app-id \
  --master-key papr-oss-master-key
```

### Qdrant Collections Missing

Collections are auto-created on startup. To manually initialize:

```bash
# Run inside container
docker exec -it papr-memory python scripts/init_qdrant_collections_opensource.py
```

### Parse Dashboard Connection Error

- Make sure Parse Dashboard is enabled: `--profile dashboard`
- Refresh the page - Parse Server may have been starting when you first accessed it
- Check Parse Server logs: `docker compose -f docker compose.yaml logs parse-server`

### Health Check Failures

```bash
# Check API health
curl http://localhost:5001/health

# Check individual service health
docker compose -f docker compose.yaml ps
```

## Stopping Services

```bash
# Stop all services
docker compose -f docker compose.yaml down

# Stop and remove volumes (WARNING: deletes all data)
docker compose -f docker compose.yaml down -v
```

## Upgrading

```bash
# Pull latest changes
git pull origin main

# Rebuild containers with latest code
docker compose -f docker compose.yaml up -d --build

# The entrypoint script will handle schema migrations automatically
```

## Open Source vs Cloud

| Feature | Open Source | Cloud (memory.papr.ai) |
|---------|-------------|------------------------|
| Core Memory API | ✅ | ✅ |
| Vector Search | ✅ | ✅ |
| Graph Relationships | ✅ | ✅ |
| Semantic Search | ✅ | ✅ |
| Batch Operations | ✅ | ✅ |
| Parse Server Auth | ✅ | ✅ |
| **Auto-Initialization** | ✅ | ✅ |
| **Self-Hosted** | ✅ | ❌ |
| **API Key Management** | Manual Script | Dashboard UI |
| **OAuth (Auth0)** | ❌ | ✅ |
| **Multi-Tenancy** | ❌ | ✅ |
| **Temporal Workflows** | ❌ | ✅ |
| **Document Processing** | ❌ | ✅ |
| **Advanced Analytics** | ❌ | ✅ |
| **GraphQL API** | ❌ | ✅ |
| **Team Collaboration** | ❌ | ✅ |
| **Priority Support** | ❌ | ✅ |
| **SLA Guarantees** | ❌ | ✅ |

## Configuration Files

- **`.env.opensource`**: Environment variables for open-source setup
- **`docker compose.yaml`**: Docker Compose configuration
- **`config/opensource.yaml`**: Feature flags and limits for open-source edition

## Need Help?

- **Documentation**: https://docs.papr.ai
- **GitHub Issues**: https://github.com/Papr-ai/memory-opensource/issues
- **Community Discord**: https://discord.gg/sWpR5a3H
- **Cloud Version**: https://memory.papr.ai (more features, managed hosting)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Open source under [LICENSE](LICENSE). Cloud features remain proprietary.
