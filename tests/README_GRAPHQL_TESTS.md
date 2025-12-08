# GraphQL Tests

Comprehensive test suite for PAPR Memory GraphQL implementation.

---

## Test Files

### 1. `test_graphql_routes.py`
Tests for the GraphQL proxy endpoint.

**Coverage**:
- ✅ Authentication (API key, session token, bearer token)
- ✅ Query forwarding to Neo4j
- ✅ Variables and operation names
- ✅ Error handling (Neo4j errors, timeouts)
- ✅ Provider credentials (NEO4J_PROVIDER_ID/KEY)
- ✅ GraphQL Playground (dev vs production)

**Run**:
```bash
pytest tests/test_graphql_routes.py -v
```

### 2. `test_graphql_multi_tenant_isolation.py`
Tests for multi-tenant security and isolation.

**Coverage**:
- ✅ JWT contains correct user_id and workspace_id
- ✅ Different users get different JWT claims
- ✅ Users cannot access other users' data
- ✅ Workspace-scoped queries filtered correctly
- ✅ Personal vs workspace data isolation
- ✅ @authorization directive compliance

**Run**:
```bash
pytest tests/test_graphql_multi_tenant_isolation.py -v
```

### 3. `test_graphql_jwt_integration.py`
Tests for JWT service and JWKS endpoint.

**Coverage**:
- ✅ JWT generation with all claims
- ✅ JWT verification and signature validation
- ✅ Token expiration handling
- ✅ JWKS endpoint format and caching
- ✅ Integration with GraphQL proxy
- ✅ Fresh JWTs for each request

**Run**:
```bash
pytest tests/test_graphql_jwt_integration.py -v
```

### 4. `test_jwt_service.py`
Unit tests for JWT service (already exists).

**Coverage**:
- ✅ JWT service initialization
- ✅ Token generation and verification
- ✅ Expired token handling
- ✅ Singleton pattern

**Run**:
```bash
pytest tests/test_jwt_service.py -v
```

---

## Environment Variables Required

Add these to your `.env` file:

```bash
# Authentication Tokens
TEST_X_PAPR_API_KEY=your_papr_api_key
TEST_X_USER_API_KEY=different_user_api_key
TEST_SESSION_TOKEN=your_session_token
TEST_BEARER_TOKEN=your_bearer_token

# Neo4j GraphQL
NEO4J_GRAPHQL_ENDPOINT=https://de7df98e-graphql.production-orch-0042.neo4j.io/graphql
NEO4J_PROVIDER_ID=your_provider_id
NEO4J_PROVIDER_KEY=your_provider_key

# JWT Keys
JWT_PRIVATE_KEY_PATH=/path/to/memory/keys/jwt-private.pem
JWT_PUBLIC_KEY_PATH=/path/to/memory/keys/jwt-public.pem
```

---

## Running Tests

### Run All GraphQL Tests

```bash
pytest tests/test_graphql*.py -v
```

### Run Specific Test Class

```bash
pytest tests/test_graphql_routes.py::TestGraphQLProxy -v
```

### Run Specific Test

```bash
pytest tests/test_graphql_routes.py::TestGraphQLProxy::test_graphql_endpoint_with_api_key -v
```

### Run with Coverage

```bash
pytest tests/test_graphql*.py --cov=routers/v1 --cov=services/jwt_service --cov-report=html
```

### Run with Detailed Output

```bash
pytest tests/test_graphql*.py -v -s
```

---

## Test Patterns

### 1. Mocking Neo4j Responses

Tests mock the Neo4j GraphQL endpoint to avoid hitting the actual server:

```python
from unittest.mock import patch, AsyncMock

with patch('httpx.AsyncClient.post') as mock_post:
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.content = json.dumps({
        "data": {"__typename": "Query"}
    }).encode()
    mock_post.return_value = mock_response

    # Make request
    response = client.post("/v1/graphql", ...)
```

### 2. Capturing JWT Tokens

Tests capture JWTs to verify claims:

```python
captured_jwt = None

def capture_jwt(*args, **kwargs):
    nonlocal captured_jwt
    auth_header = kwargs["headers"].get("Authorization", "")
    if auth_header.startswith("Bearer "):
        captured_jwt = auth_header.replace("Bearer ", "")

    # Return mock response
    ...

with patch('httpx.AsyncClient.post', side_effect=capture_jwt):
    # Make request
    ...

    # Verify JWT
    decoded = pyjwt.decode(captured_jwt, options={"verify_signature": False})
    assert decoded["user_id"] == "expected_user_id"
```

### 3. Testing Multi-Tenant Isolation

Tests verify that different users get different JWTs:

```python
# User A request
with patch(...):
    response_a = client.post("/v1/graphql", headers={"X-API-Key": USER_A_KEY})

# User B request
with patch(...):
    response_b = client.post("/v1/graphql", headers={"X-API-Key": USER_B_KEY})

# Different users should have different claims
assert decoded_a["user_id"] != decoded_b["user_id"]
```

---

## Test Data

### Code Schema Types

Tests use types from `models/papr_graphql_code_schema.py`:

- `CodeProject`
- `CodeSnippet`
- `Developer`
- `Function`
- `Library`
- `Module`

### Example Queries

**Get Code Projects:**
```graphql
query GetCodeProjects {
  codeProjects(options: { limit: 5 }) {
    id
    name
    description
  }
}
```

**Get Code Project by ID:**
```graphql
query GetCodeProject($id: ID!) {
  codeProject(id: $id) {
    id
    name
    user_id
    workspace_id
  }
}
```

---

## Troubleshooting

### Issue: "TEST_X_PAPR_API_KEY not set"

**Solution**: Add your API key to `.env`:
```bash
TEST_X_PAPR_API_KEY=your_actual_api_key
```

### Issue: "JWT private key not found"

**Solution**: Generate RSA keys:
```bash
cd /Users/shawkatkabbara/Documents/GitHub/memory
mkdir -p keys
openssl genrsa -out keys/jwt-private.pem 2048
openssl rsa -in keys/jwt-private.pem -pubout -out keys/jwt-public.pem
```

### Issue: Tests fail with "Connection refused"

**Solution**: Tests mock Neo4j responses - no actual connection needed. Make sure mocking is working:
```python
with patch('httpx.AsyncClient.post') as mock_post:
    # Ensure mock_post is being called
    ...
```

### Issue: "Invalid signature" when verifying JWT

**Solution**: Make sure public and private keys match:
```bash
# Regenerate both keys together
openssl genrsa -out keys/jwt-private.pem 2048
openssl rsa -in keys/jwt-private.pem -pubout -out keys/jwt-public.pem
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: GraphQL Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov

      - name: Generate test keys
        run: |
          mkdir -p keys
          openssl genrsa -out keys/jwt-private.pem 2048
          openssl rsa -in keys/jwt-private.pem -pubout -out keys/jwt-public.pem

      - name: Run GraphQL tests
        env:
          TEST_X_PAPR_API_KEY: ${{ secrets.TEST_API_KEY }}
          NEO4J_PROVIDER_ID: ${{ secrets.NEO4J_PROVIDER_ID }}
          NEO4J_PROVIDER_KEY: ${{ secrets.NEO4J_PROVIDER_KEY }}
        run: |
          pytest tests/test_graphql*.py -v --cov=routers/v1 --cov=services/jwt_service
```

---

## Test Coverage Goals

- ✅ **Authentication**: All auth methods tested (API key, bearer, session)
- ✅ **JWT Generation**: Token format, claims, expiration
- ✅ **JWT Verification**: Signature validation, expiration checking
- ✅ **JWKS Endpoint**: Format, caching, CORS headers
- ✅ **Multi-Tenant Isolation**: User separation, workspace filtering
- ✅ **Error Handling**: Timeouts, invalid queries, Neo4j errors
- ✅ **Neo4j Integration**: Provider credentials, query forwarding

---

## Next Steps

1. **Run Tests Locally**: Ensure all tests pass
2. **Add Integration Tests**: Test with real Neo4j endpoint (optional)
3. **Performance Tests**: Test query performance under load
4. **Security Audit**: Review multi-tenant isolation thoroughly
5. **CI Integration**: Add tests to GitHub Actions

---

**Version**: 1.0
**Last Updated**: 2025-10-30
