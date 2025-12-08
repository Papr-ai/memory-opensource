# GraphQL Setup Checklist

Use this checklist to set up GraphQL for PAPR Memory.

---

## ✅ Prerequisites

- [ ] RSA key pair generated (`keys/jwt-private.pem` and `keys/jwt-public.pem`)
- [ ] Environment variables configured
- [ ] Dependencies installed

---

## 🔑 Step 1: Environment Variables

Add these to your `.env` file:

```bash
# Neo4j GraphQL Endpoint
NEO4J_GRAPHQL_ENDPOINT=https://de7df98e-graphql.production-orch-0042.neo4j.io/graphql
NEO4J_PROVIDER_ID=your_provider_id
NEO4J_PROVIDER_KEY=your_provider_key

# JWT Keys (paths to your generated keys)
JWT_PRIVATE_KEY_PATH=/path/to/memory/keys/jwt-private.pem
JWT_PUBLIC_KEY_PATH=/path/to/memory/keys/jwt-public.pem

# Optional: Environment
ENVIRONMENT=development  # or production
```

---

## 📦 Step 2: Install Dependencies

```bash
cd /Users/shawkatkabbara/Documents/GitHub/memory

# Install new dependencies
pip install PyJWT==2.8.0 cryptography==41.0.7

# Or install all
pip install -r requirements.txt
```

---

## 🔐 Step 3: Verify RSA Keys

```bash
# Check that keys exist
ls -la keys/

# You should see:
# jwt-private.pem  (private key - keep secret!)
# jwt-public.pem   (public key - can be shared)
```

**Important**: Make sure `keys/jwt-private.pem` is in `.gitignore`!

```bash
# Verify it's ignored
cat .gitignore | grep jwt-private.pem
```

---

## 🧪 Step 4: Test JWT Service

```bash
# Run the JWT service test
cd /Users/shawkatkabbara/Documents/GitHub/memory

python -c "
from services.jwt_service import get_jwt_service

jwt_service = get_jwt_service()
token = jwt_service.generate_token(
    user_id='test_user',
    workspace_id='test_workspace'
)

print('✅ JWT generated successfully!')
print(f'Token: {token[:50]}...')

# Verify it
payload = jwt_service.verify_token(token)
print(f'✅ JWT verified! User ID: {payload[\"user_id\"]}')
"
```

Expected output:
```
✅ JWT generated successfully!
Token: eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOi...
✅ JWT verified! User ID: test_user
```

---

## 🌐 Step 5: Register Routes

Make sure the routes are registered in your FastAPI app.

**File**: `/Users/shawkatkabbara/Documents/GitHub/memory/main.py` (or wherever you set up routes)

```python
from routers.v1 import graphql_routes, jwks_routes

# Register JWKS endpoint (for Neo4j to validate JWTs)
app.include_router(jwks_routes.router)

# Register GraphQL proxy endpoint
app.include_router(graphql_routes.router, prefix="/v1")
```

---

## 🚀 Step 6: Start the Server

```bash
# Start FastAPI server
cd /Users/shawkatkabbara/Documents/GitHub/memory

python -m uvicorn main:app --reload --port 8000
```

---

## ✓ Step 7: Test JWKS Endpoint

```bash
# Test JWKS endpoint (should return public key)
curl http://localhost:8000/.well-known/jwks.json

# Expected response:
# {
#   "keys": [
#     {
#       "kty": "RSA",
#       "use": "sig",
#       "kid": "papr-memory-key-1",
#       "alg": "RS256",
#       "n": "...",
#       "e": "AQAB"
#     }
#   ]
# }
```

---

## ✓ Step 8: Test GraphQL Proxy

### Test 1: Introspection Query

```bash
curl -X POST http://localhost:8000/v1/graphql \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "query { __typename }"}'
```

Expected: `{"data": {"__typename": "Query"}}`

### Test 2: Real Query (if schema is set up)

```bash
curl -X POST http://localhost:8000/v1/graphql \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query { tasks(options: { limit: 5 }) { id title status } }"
  }'
```

---

## ✓ Step 9: Test GraphQL Playground

Open in browser:
```
http://localhost:8000/v1/graphql
```

Enter your API key when prompted, then try a query:

```graphql
query GetTasks {
  tasks(options: { limit: 5 }) {
    id
    title
    status
  }
}
```

---

## 🔧 Step 10: Configure Neo4j Schema

Your Neo4j GraphQL schema needs to be updated with authorization directives.

**Update**: `/Users/shawkatkabbara/Documents/GitHub/memory/models/papr_graphql_default_schema.py`

Add at the top:

```graphql
extend schema
  @authentication(
    type: JWT
    jwks: {
      url: "https://memory.papr.ai/.well-known/jwks.json"
    }
  )
```

For each type, add `@authorization` directive:

```graphql
type Task @node
  @authorization(
    validate: [
      {
        when: [BEFORE]
        operations: [READ, UPDATE, DELETE, CREATE]
        where: { node: { user_id: "$jwt.user_id" } }
      }
    ]
  )
{
  id: String!
  title: String!
  user_id: String!
  workspace_id: String
  # ... other fields
}
```

**Upload to Neo4j**: Use Neo4j's admin API or console to update the schema.

---

## ✅ Verification Checklist

- [ ] JWT service generates tokens successfully
- [ ] JWKS endpoint returns public key
- [ ] GraphQL proxy endpoint responds to requests
- [ ] Authentication works (API key → JWT)
- [ ] GraphQL Playground loads
- [ ] Queries execute and return data
- [ ] Multi-tenant filtering works (users only see their data)

---

## 🐛 Troubleshooting

### Issue: "FileNotFoundError: JWT private key not found"

**Solution**: Make sure you generated the RSA keys:

```bash
cd /Users/shawkatkabbara/Documents/GitHub/memory
mkdir -p keys
openssl genrsa -out keys/jwt-private.pem 2048
openssl rsa -in keys/jwt-private.pem -pubout -out keys/jwt-public.pem
```

### Issue: "Missing authentication"

**Solution**: Make sure you're sending `X-API-Key` header in your request.

### Issue: JWKS endpoint returns 500

**Solution**: Check that the public key file exists and is readable:

```bash
ls -la keys/jwt-public.pem
```

### Issue: "Neo4j GraphQL error"

**Solutions**:
1. Check `NEO4J_GRAPHQL_ENDPOINT` is correct
2. Verify `NEO4J_PROVIDER_ID` and `NEO4J_PROVIDER_KEY` are set
3. Check Neo4j logs for authentication errors

### Issue: "Forbidden" even with valid auth

**Solution**:
1. Verify JWT contains `user_id` and `workspace_id` claims
2. Check that Neo4j schema has `@authorization` directives
3. Verify JWKS endpoint is publicly accessible (Neo4j must be able to reach it)

---

## 📚 Next Steps

After completing this checklist:

1. **Read the Architecture**: `docs/roadmap/graphql/graphql-architecture.md`
2. **Try Example Queries**: `docs/roadmap/graphql/quickstart.md`
3. **Integrate with SDK**: Update Python and TypeScript SDKs
4. **Deploy to Staging**: Test with real data
5. **Security Audit**: Verify multi-tenant isolation
6. **Production Deployment**: Update environment variables and deploy

---

## 📞 Support

If you encounter issues:

1. Check the logs: `tail -f logs/app.log`
2. Review Neo4j GraphQL logs
3. Test JWT generation separately
4. Verify environment variables are loaded

---

## 🧪 Step 11: Run Tests

```bash
cd /Users/shawkatkabbara/Documents/GitHub/memory

# Run all GraphQL tests
pytest tests/test_graphql*.py -v

# Run specific test suites
pytest tests/test_graphql_routes.py -v
pytest tests/test_graphql_multi_tenant_isolation.py -v
pytest tests/test_graphql_jwt_integration.py -v

# Run with coverage
pytest tests/test_graphql*.py --cov=routers/v1 --cov=services/jwt_service --cov-report=html
```

**Expected Results**:
- ✅ All authentication tests pass
- ✅ JWT generation and verification works
- ✅ JWKS endpoint returns valid format
- ✅ Multi-tenant isolation is enforced
- ✅ Neo4j provider credentials are sent

See `tests/README_GRAPHQL_TESTS.md` for detailed test documentation.

---

**Status**: Setup Complete ✅
**Next**: Configure Neo4j Schema with @authorization directives
