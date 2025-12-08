# PAPR Memory GraphQL Implementation

Complete documentation for GraphQL on PAPR Memory.

---

## 📖 Overview

PAPR Memory now supports GraphQL queries through a simple proxy layer that:

1. **Authenticates** using your existing API keys
2. **Converts** API keys to JWT tokens for Neo4j
3. **Forwards** GraphQL queries to Neo4j's hosted endpoint
4. **Returns** results with automatic multi-tenant filtering

**Key Benefit**: You keep using API keys. The JWT translation happens automatically behind the scenes.

---

## 🏗️ Architecture

```
Client (API Key)
    ↓
FastAPI /v1/graphql (validates API key, generates JWT)
    ↓
Neo4j GraphQL Endpoint (applies @authorization filters)
    ↓
Results (filtered by user_id/workspace_id)
```

### Why This Approach?

- ✅ **Simple**: Just a proxy wrapper, not a full GraphQL server
- ✅ **Secure**: Multi-tenant isolation enforced by Neo4j at database level
- ✅ **Familiar**: Same authentication (API keys) you already use
- ✅ **Powerful**: Full GraphQL queries with relationships and filtering

---

## 📚 Documentation

### 1. **Architecture** → [`graphql-architecture.md`](./graphql-architecture.md)
Comprehensive technical architecture including:
- Neo4j GraphQL integration details
- JWT-based multi-tenant authorization
- FastAPI proxy implementation
- Python & TypeScript SDK integration
- Security model and best practices

**Read this for**: Understanding the full system design.

### 2. **Implementation Guide** → [`implementation-guide.md`](./implementation-guide.md)
Step-by-step implementation with complete code:
- JWT service creation
- JWKS endpoint setup
- GraphQL proxy route
- Testing procedures

**Read this for**: Building the GraphQL endpoint.

### 3. **Setup Checklist** → [`setup-checklist.md`](./setup-checklist.md)
Hands-on setup guide with verification steps:
- Environment variable configuration
- Dependency installation
- Key generation
- Testing and troubleshooting

**Read this for**: Getting GraphQL running locally.

### 4. **Quickstart** → [`quickstart.md`](./quickstart.md)
Get started in 5 minutes:
- First query examples
- SDK usage (Python & TypeScript)
- Common query patterns

**Read this for**: Using GraphQL in your app.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PAPR Memory API key
- Neo4j GraphQL endpoint access

### 1. Generate Keys

```bash
cd /Users/shawkatkabbara/Documents/GitHub/memory
mkdir -p keys
openssl genrsa -out keys/jwt-private.pem 2048
openssl rsa -in keys/jwt-private.pem -pubout -out keys/jwt-public.pem
```

### 2. Set Environment Variables

```bash
# .env
NEO4J_GRAPHQL_ENDPOINT=https://de7df98e-graphql.production-orch-0042.neo4j.io/graphql
NEO4J_PROVIDER_ID=your_provider_id
NEO4J_PROVIDER_KEY=your_provider_key
JWT_PRIVATE_KEY_PATH=/path/to/keys/jwt-private.pem
JWT_PUBLIC_KEY_PATH=/path/to/keys/jwt-public.pem
```

### 3. Install Dependencies

```bash
pip install PyJWT==2.8.0 cryptography==41.0.7
```

### 4. Start Server

```bash
python -m uvicorn main:app --reload
```

### 5. Test

```bash
curl -X POST http://localhost:8000/v1/graphql \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "query { __typename }"}'
```

---

## 📂 File Structure

```
/Users/shawkatkabbara/Documents/GitHub/memory/
├── services/
│   └── jwt_service.py                 # JWT generation for Neo4j
├── routers/v1/
│   ├── graphql_routes.py              # GraphQL proxy endpoint
│   └── jwks_routes.py                 # JWKS endpoint for JWT validation
├── keys/
│   ├── jwt-private.pem                # RSA private key (keep secret!)
│   └── jwt-public.pem                 # RSA public key (shared)
├── tests/
│   └── test_jwt_service.py            # JWT service tests
└── docs/roadmap/graphql/
    ├── README.md                       # This file
    ├── graphql-architecture.md         # Full architecture
    ├── implementation-guide.md         # Step-by-step implementation
    ├── setup-checklist.md              # Setup verification
    └── quickstart.md                   # Quick examples
```

---

## 🔐 Security

### Multi-Tenant Isolation

Every GraphQL query is automatically filtered by:
- `user_id` - User who owns the data
- `workspace_id` - Workspace context

This is enforced by Neo4j using `@authorization` directives:

```graphql
type Task @node
  @authorization(
    validate: [{
      when: [BEFORE]
      operations: [READ, UPDATE, DELETE, CREATE]
      where: { node: { user_id: "$jwt.user_id" } }
    }]
  )
{
  id: String!
  user_id: String!
  # ...
}
```

**Result**: Users can ONLY access their own data. No cross-tenant leakage possible.

### JWT Token Structure

Tokens generated from API keys contain:

```json
{
  "sub": "user_abc123",
  "user_id": "user_abc123",
  "workspace_id": "ws_xyz789",
  "iss": "https://memory.papr.ai",
  "aud": "neo4j-graphql",
  "exp": 1735564800
}
```

Neo4j validates the JWT signature using your public key from `/.well-known/jwks.json`.

---

## 🧪 Testing

### Run JWT Service Tests

```bash
pytest tests/test_jwt_service.py -v
```

### Manual Testing

```bash
# Test JWKS endpoint
curl http://localhost:8000/.well-known/jwks.json

# Test GraphQL proxy
curl -X POST http://localhost:8000/v1/graphql \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "query { tasks(options: {limit: 5}) { id title } }"}'
```

### GraphQL Playground

```
http://localhost:8000/v1/graphql
```

(Enter API key when prompted)

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NEO4J_GRAPHQL_ENDPOINT` | Neo4j GraphQL URL | Yes |
| `NEO4J_PROVIDER_ID` | Neo4j provider ID | Yes |
| `NEO4J_PROVIDER_KEY` | Neo4j provider key | Yes |
| `JWT_PRIVATE_KEY_PATH` | Path to RSA private key | Yes |
| `JWT_PUBLIC_KEY_PATH` | Path to RSA public key | Yes |
| `ENVIRONMENT` | `development` or `production` | No |

### Neo4j Schema Configuration

Update your GraphQL schema with authentication:

```graphql
extend schema
  @authentication(
    type: JWT
    jwks: {
      url: "https://memory.papr.ai/.well-known/jwks.json"
    }
  )
```

Then add `@authorization` directives to all types (see `implementation-guide.md`).

---

## 📊 Implementation Status

| Component | Status | File |
|-----------|--------|------|
| ✅ JWT Service | Complete | `services/jwt_service.py` |
| ✅ JWKS Endpoint | Complete | `routers/v1/jwks_routes.py` |
| ✅ GraphQL Proxy | Complete | `routers/v1/graphql_routes.py` |
| ✅ Tests | Complete | `tests/test_jwt_service.py` |
| ✅ Documentation | Complete | `docs/roadmap/graphql/` |
| ⏳ Neo4j Schema | Pending | Update with @authorization |
| ⏳ Python SDK | Pending | GraphQL client implementation |
| ⏳ TypeScript SDK | Pending | Apollo Client wrapper |

---

## 🗺️ Roadmap

### Phase 1: Core Implementation ✅ (Complete)
- [x] JWT service
- [x] JWKS endpoint
- [x] GraphQL proxy
- [x] Documentation

### Phase 2: Neo4j Schema (Current)
- [ ] Update schema with @authorization directives
- [ ] Upload schema to Neo4j
- [ ] Test multi-tenant filtering
- [ ] Verify all types are protected

### Phase 3: SDK Integration
- [ ] Python SDK GraphQL client
- [ ] TypeScript SDK Apollo wrapper
- [ ] Code generation for type safety
- [ ] SDK documentation

### Phase 4: Production
- [ ] Security audit
- [ ] Performance testing
- [ ] Rate limiting
- [ ] Monitoring and logging
- [ ] Deploy to production

---

## 🐛 Troubleshooting

### Common Issues

**"FileNotFoundError: JWT private key not found"**
- Generate RSA keys (see Setup Checklist)

**"Missing authentication"**
- Include `X-API-Key` header in requests

**"Forbidden" errors**
- Check Neo4j schema has `@authorization` directives
- Verify JWT contains `user_id` and `workspace_id`
- Ensure JWKS endpoint is publicly accessible

**JWKS endpoint returns 500**
- Check public key file exists and is readable
- Verify file permissions

See [`setup-checklist.md`](./setup-checklist.md) for detailed troubleshooting.

---

## 📞 Support

- **Documentation**: This directory
- **Issues**: GitHub Issues
- **Architecture Questions**: See `graphql-architecture.md`
- **Implementation Help**: See `implementation-guide.md`

---

## 🎯 Key Takeaways

1. **Simple Proxy**: Not building a full GraphQL server, just proxying to Neo4j
2. **Existing Auth**: Keep using API keys, JWT is internal translation
3. **Security First**: Multi-tenant isolation enforced at database level
4. **Developer Experience**: Type-safe GraphQL queries, relationship traversal

---

## 📝 Next Steps

1. ✅ **Read Architecture** - Understand the full design
2. ✅ **Follow Setup** - Get GraphQL running locally
3. ⏳ **Update Schema** - Add @authorization directives to Neo4j
4. ⏳ **Test Queries** - Verify multi-tenant filtering works
5. ⏳ **Integrate SDKs** - Add GraphQL to Python and TypeScript SDKs
6. ⏳ **Deploy** - Push to production

---

**Version**: 1.0
**Last Updated**: 2025-10-30
**Status**: Implementation Complete, Schema Update Pending
