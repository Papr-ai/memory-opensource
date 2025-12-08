# GraphQL Implementation Guide - Simplified Approach

**Status**: Implementation Ready
**Last Updated**: 2025-10-30

---

## Overview

The `/v1/graphql` endpoint is a **simple proxy wrapper** on top of Neo4j's hosted GraphQL endpoint. Here's what it does:

```
Client Request (API Key)
    ↓
FastAPI /v1/graphql
    ↓
1. Authenticate using existing auth system (API key → user_id, workspace_id)
2. Convert to JWT token (new step we need to add)
3. Forward GraphQL query to Neo4j with JWT
    ↓
Neo4j GraphQL (applies @authorization filters)
    ↓
Return results
```

## What We Need to Add

Since your current system uses **API keys** (not JWT), we need to add:

1. **JWT Generation** - Convert API key → JWT token
2. **JWKS Endpoint** - So Neo4j can validate our JWTs
3. **Simple Proxy Route** - Forward requests to Neo4j

**Important**: We're NOT changing your existing authentication. We're just adding JWT as a "translation layer" for Neo4j.

---

## Step 1: Generate RSA Key Pair

Neo4j validates JWTs using public key cryptography. Generate a key pair:

```bash
# Navigate to your memory repo
cd /Users/shawkatkabbara/Documents/GitHub/memory

# Create keys directory
mkdir -p keys

# Generate private key
openssl genrsa -out keys/jwt-private.pem 2048

# Generate public key
openssl rsa -in keys/jwt-private.pem -pubout -out keys/jwt-public.pem

# Add to .gitignore (keep private key secret!)
echo "keys/jwt-private.pem" >> .gitignore
```

**Store private key securely**:
```bash
# In production, use environment variable or secrets manager
export JWT_PRIVATE_KEY_PATH="/path/to/keys/jwt-private.pem"
export JWT_PUBLIC_KEY_PATH="/path/to/keys/jwt-public.pem"
```

---

## Step 2: Create JWT Service

This converts your existing auth response → JWT token for Neo4j.

**File**: `/Users/shawkatkabbara/Documents/GitHub/memory/services/jwt_service.py`

```python
"""
JWT Service for Neo4j GraphQL Authentication

Converts PAPR Memory's existing authentication (API keys, bearer tokens, session tokens)
into JWT tokens that Neo4j GraphQL can validate.
"""

from datetime import datetime, timedelta, UTC
from typing import Optional
import jwt
import os
from pathlib import Path

class JWTService:
    """
    Generate JWT tokens for Neo4j GraphQL from existing auth.

    This is a translation layer - we keep using API keys for client auth,
    but generate JWTs for Neo4j's authorization directives.
    """

    def __init__(self):
        self.algorithm = "RS256"  # RSA signing
        self.issuer = "https://memory.papr.ai"
        self.audience = "neo4j-graphql"

        # Load private key for signing JWTs
        private_key_path = os.getenv(
            "JWT_PRIVATE_KEY_PATH",
            str(Path(__file__).parent.parent / "keys" / "jwt-private.pem")
        )

        with open(private_key_path, 'r') as f:
            self.private_key = f.read()

    def generate_token(
        self,
        user_id: str,
        workspace_id: Optional[str] = None,
        end_user_id: Optional[str] = None,
        roles: Optional[list[str]] = None,
        expires_in_minutes: int = 60
    ) -> str:
        """
        Generate a JWT token for Neo4j GraphQL authorization.

        Args:
            user_id: User's unique identifier (required for @authorization filters)
            workspace_id: Workspace identifier (optional, for workspace-level data)
            end_user_id: End user identifier (optional)
            roles: User roles (optional, for role-based access)
            expires_in_minutes: Token expiration time

        Returns:
            Signed JWT token string

        Example:
            jwt_service = JWTService()
            token = jwt_service.generate_token(
                user_id="user_abc123",
                workspace_id="ws_xyz789"
            )
        """
        now = datetime.now(UTC)
        expiration = now + timedelta(minutes=expires_in_minutes)

        # Build JWT payload with claims Neo4j will use in @authorization directives
        payload = {
            "sub": user_id,              # Standard JWT subject claim
            "user_id": user_id,          # Custom claim for @authorization
            "workspace_id": workspace_id, # Custom claim for @authorization
            "end_user_id": end_user_id,
            "roles": roles or [],
            "iss": self.issuer,          # Issuer
            "aud": self.audience,        # Audience
            "exp": int(expiration.timestamp()),  # Expiration
            "iat": int(now.timestamp())   # Issued at
        }

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        # Sign and return token
        token = jwt.encode(
            payload,
            self.private_key,
            algorithm=self.algorithm
        )

        return token

    def verify_token(self, token: str) -> dict:
        """
        Verify a JWT token (for testing purposes).

        In production, Neo4j will verify tokens using our JWKS endpoint.
        This method is just for local testing.
        """
        # Load public key
        public_key_path = os.getenv(
            "JWT_PUBLIC_KEY_PATH",
            str(Path(__file__).parent.parent / "keys" / "jwt-public.pem")
        )

        with open(public_key_path, 'r') as f:
            public_key = f.read()

        # Verify and decode
        payload = jwt.decode(
            token,
            public_key,
            algorithms=[self.algorithm],
            audience=self.audience,
            issuer=self.issuer
        )

        return payload


# Singleton instance
_jwt_service = None

def get_jwt_service() -> JWTService:
    """Get singleton JWT service instance"""
    global _jwt_service
    if _jwt_service is None:
        _jwt_service = JWTService()
    return _jwt_service
```

---

## Step 3: Create JWKS Endpoint

Neo4j needs to fetch your public key to validate JWTs.

**File**: `/Users/shawkatkabbara/Documents/GitHub/memory/routers/v1/jwks_routes.py`

```python
"""
JWKS (JSON Web Key Set) Endpoint

Provides public keys for Neo4j GraphQL to validate JWT signatures.
"""

from fastapi import APIRouter, Response
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import json
import base64
import os
from pathlib import Path

router = APIRouter(tags=["JWKS"])

@router.get("/.well-known/jwks.json")
async def get_jwks():
    """
    JWKS endpoint for Neo4j GraphQL JWT validation.

    Neo4j will call this endpoint to fetch public keys for verifying JWTs.
    This is a standard OAuth2/OpenID Connect endpoint.

    Returns:
        JSON Web Key Set (JWKS) with public keys
    """

    # Load public key
    public_key_path = os.getenv(
        "JWT_PUBLIC_KEY_PATH",
        str(Path(__file__).parent.parent.parent / "keys" / "jwt-public.pem")
    )

    with open(public_key_path, 'rb') as f:
        public_key = serialization.load_pem_public_key(
            f.read(),
            backend=default_backend()
        )

    # Extract RSA public numbers
    public_numbers = public_key.public_numbers()

    def encode_int_base64url(n: int) -> str:
        """Encode integer as base64url (JWK format)"""
        # Convert to bytes (big-endian)
        byte_length = (n.bit_length() + 7) // 8
        n_bytes = n.to_bytes(byte_length, byteorder='big')

        # Base64url encode (no padding)
        encoded = base64.urlsafe_b64encode(n_bytes).rstrip(b'=').decode('utf-8')
        return encoded

    # Build JWK (JSON Web Key) in standard format
    jwk = {
        "kty": "RSA",                                    # Key type
        "use": "sig",                                     # Usage: signature
        "kid": "papr-memory-key-1",                      # Key ID
        "alg": "RS256",                                   # Algorithm
        "n": encode_int_base64url(public_numbers.n),     # Modulus
        "e": encode_int_base64url(public_numbers.e),     # Exponent
    }

    # JWKS format (set of keys)
    jwks = {
        "keys": [jwk]
    }

    return Response(
        content=json.dumps(jwks, indent=2),
        media_type="application/json",
        headers={
            "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
            "Access-Control-Allow-Origin": "*",       # Allow Neo4j to fetch
        }
    )
```

**Register this route** in your main app:

```python
# In /Users/shawkatkabbara/Documents/GitHub/memory/main.py or wherever you set up routes

from routers.v1 import jwks_routes

app.include_router(jwks_routes.router)
```

---

## Step 4: Create GraphQL Proxy Route

This is the simple wrapper that ties everything together.

**File**: `/Users/shawkatkabbara/Documents/GitHub/memory/routers/v1/graphql_routes.py`

```python
"""
GraphQL Proxy Route

Simple wrapper that:
1. Authenticates using existing auth system (API keys, bearer tokens, session tokens)
2. Converts auth to JWT token
3. Forwards GraphQL query to Neo4j with JWT
4. Returns response
"""

from fastapi import APIRouter, Request, Depends, Response, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
from typing import Optional, Dict, Any
import httpx
import time

from services.auth_utils import get_user_from_token_optimized
from services.jwt_service import get_jwt_service
from services.logger_singleton import LoggerSingleton
from memory.memory_graph import MemoryGraph
from services.utils import get_memory_graph

router = APIRouter(prefix="/graphql", tags=["GraphQL"])

# Security schemes (reuse existing ones)
bearer_auth = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
session_token_header = APIKeyHeader(name="X-Session-Token", auto_error=False)

# Neo4j GraphQL endpoint
NEO4J_GRAPHQL_ENDPOINT = "https://de7df98e-graphql.production-orch-0042.neo4j.io/graphql"

logger = LoggerSingleton.get_logger(__name__)

@router.post("",
    description="""
    GraphQL endpoint for querying PAPR Memory using GraphQL.

    This endpoint proxies GraphQL queries to Neo4j's hosted GraphQL endpoint,
    automatically applying multi-tenant authorization filters based on user_id and workspace_id.

    **Authentication Required**:
    One of the following authentication methods must be used:
    - Bearer token in `Authorization` header
    - API Key in `X-API-Key` header
    - Session token in `X-Session-Token` header

    **Request Body**:
    ```json
    {
      "query": "query { project(id: \\"proj_123\\") { name tasks { title } } }",
      "variables": {},
      "operationName": "GetProject"
    }
    ```

    **Example Query**:
    ```graphql
    query GetProjectTasks($projectId: ID!) {
      project(id: $projectId) {
        name
        tasks {
          title
          status
        }
      }
    }
    ```

    All queries are automatically filtered by user_id and workspace_id for security.
    """,
    openapi_extra={
        "operationId": "graphql_query_v1",
        "x-openai-isConsequential": False
    }
)
async def graphql_proxy(
    request: Request,
    response: Response,
    api_key: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Depends(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
):
    """
    GraphQL proxy endpoint.

    Flow:
    1. Authenticate using existing system (API key, bearer token, session token)
    2. Extract user_id and workspace_id
    3. Generate JWT token with these claims
    4. Forward GraphQL query to Neo4j with JWT
    5. Return response
    """

    start_time = time.time()

    # --- Step 1: Authenticate (using existing auth system) ---
    client_type = request.headers.get('X-Client-Type', 'graphql_client')

    logger.info("GraphQL request received", extra={
        "client_type": client_type,
        "has_api_key": bool(api_key),
        "has_bearer": bool(bearer_token),
        "has_session": bool(session_token)
    })

    try:
        async with httpx.AsyncClient() as httpx_client:
            # Reuse existing authentication logic
            if api_key and bearer_token:
                auth_response = await get_user_from_token_optimized(
                    f"Bearer {bearer_token.credentials}",
                    client_type,
                    memory_graph,
                    api_key=api_key,
                    httpx_client=httpx_client
                )
            elif api_key:
                auth_response = await get_user_from_token_optimized(
                    f"APIKey {api_key}",
                    client_type,
                    memory_graph,
                    httpx_client=httpx_client
                )
            elif bearer_token:
                auth_response = await get_user_from_token_optimized(
                    f"Bearer {bearer_token.credentials}",
                    client_type,
                    memory_graph,
                    httpx_client=httpx_client
                )
            elif session_token:
                auth_response = await get_user_from_token_optimized(
                    f"Session {session_token}",
                    client_type,
                    memory_graph,
                    httpx_client=httpx_client
                )
            else:
                raise HTTPException(401, "Missing authentication")

    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(401, f"Authentication failed: {str(e)}")

    if not auth_response:
        raise HTTPException(401, "Invalid authentication")

    # Extract user information
    user_id = auth_response.developer_id
    workspace_id = auth_response.workspace_id
    end_user_id = auth_response.end_user_id

    logger.info("User authenticated", extra={
        "user_id": user_id,
        "workspace_id": workspace_id,
        "end_user_id": end_user_id
    })

    # --- Step 2: Generate JWT for Neo4j ---
    jwt_service = get_jwt_service()

    try:
        neo4j_jwt = jwt_service.generate_token(
            user_id=user_id,
            workspace_id=workspace_id,
            end_user_id=end_user_id,
            roles=["developer"],  # Can be expanded based on user permissions
            expires_in_minutes=60
        )
    except Exception as e:
        logger.error(f"JWT generation failed: {e}")
        raise HTTPException(500, f"JWT generation failed: {str(e)}")

    # --- Step 3: Parse GraphQL request ---
    try:
        body = await request.json()
        query = body.get("query")
        variables = body.get("variables", {})
        operation_name = body.get("operationName")

        if not query:
            raise ValueError("Missing 'query' field in request body")

    except Exception as e:
        logger.error(f"Invalid GraphQL request: {e}")
        raise HTTPException(400, f"Invalid GraphQL request: {str(e)}")

    logger.info("GraphQL query", extra={
        "operation_name": operation_name,
        "variables": variables,
        "query_length": len(query)
    })

    # --- Step 4: Forward to Neo4j GraphQL ---
    async with httpx.AsyncClient() as client:
        try:
            neo4j_response = await client.post(
                NEO4J_GRAPHQL_ENDPOINT,
                json={
                    "query": query,
                    "variables": variables,
                    "operationName": operation_name
                },
                headers={
                    "Authorization": f"Bearer {neo4j_jwt}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )

            duration = time.time() - start_time

            logger.info("GraphQL query completed", extra={
                "status_code": neo4j_response.status_code,
                "duration_ms": duration * 1000,
                "user_id": user_id
            })

            # Return Neo4j's response as-is
            return Response(
                content=neo4j_response.content,
                status_code=neo4j_response.status_code,
                media_type="application/json"
            )

        except httpx.TimeoutException:
            logger.error("Neo4j GraphQL timeout")
            raise HTTPException(504, "Neo4j GraphQL request timeout")
        except httpx.HTTPStatusError as e:
            logger.error(f"Neo4j GraphQL error: {e.response.status_code}")
            raise HTTPException(e.response.status_code, f"Neo4j GraphQL error: {e.response.text}")
        except Exception as e:
            logger.error(f"Neo4j GraphQL request failed: {e}")
            raise HTTPException(500, f"Neo4j GraphQL error: {str(e)}")


@router.get("",
    description="GraphQL Playground (development only)"
)
async def graphql_playground():
    """
    Serve GraphQL Playground for development.
    Disabled in production for security.
    """
    import os

    if os.getenv("ENVIRONMENT") == "production":
        raise HTTPException(404, "GraphQL Playground disabled in production")

    # Simple GraphiQL playground
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>PAPR Memory GraphQL</title>
        <link rel="stylesheet" href="https://unpkg.com/graphiql/graphiql.min.css" />
        <style>
            body { margin: 0; }
            #graphiql { height: 100vh; }
        </style>
    </head>
    <body>
        <div id="graphiql"></div>

        <script crossorigin src="https://unpkg.com/react/umd/react.production.min.js"></script>
        <script crossorigin src="https://unpkg.com/react-dom/umd/react-dom.production.min.js"></script>
        <script crossorigin src="https://unpkg.com/graphiql/graphiql.min.js"></script>

        <script>
            const apiKey = prompt('Enter your API key:');

            const fetcher = GraphiQL.createFetcher({
                url: '/v1/graphql',
                headers: {
                    'X-API-Key': apiKey
                }
            });

            ReactDOM.render(
                React.createElement(GraphiQL, { fetcher: fetcher }),
                document.getElementById('graphiql'),
            );
        </script>
    </body>
    </html>
    """

    return Response(content=html, media_type="text/html")
```

**Register the route** in your v1 router:

```python
# In /Users/shawkatkabbara/Documents/GitHub/memory/routers/v1/__init__.py

from routers.v1 import graphql_routes

# Add to your v1_router
v1_router.include_router(graphql_routes.router)
```

---

## Step 5: Update Dependencies

Add JWT library to your requirements:

```bash
# Add to requirements.txt
PyJWT==2.8.0
cryptography==41.0.7
```

Install:
```bash
pip install PyJWT==2.8.0 cryptography==41.0.7
```

---

## Step 6: Test the Implementation

### Test 1: Generate JWT

```python
# test_jwt.py
from services.jwt_service import get_jwt_service

jwt_service = get_jwt_service()

token = jwt_service.generate_token(
    user_id="user_test123",
    workspace_id="ws_test456"
)

print("Generated JWT:", token)

# Verify it
payload = jwt_service.verify_token(token)
print("Decoded payload:", payload)
```

### Test 2: Test JWKS Endpoint

```bash
# Start your FastAPI server
python -m uvicorn main:app --reload

# In another terminal, test JWKS endpoint
curl http://localhost:8000/.well-known/jwks.json
```

Should return:
```json
{
  "keys": [
    {
      "kty": "RSA",
      "use": "sig",
      "kid": "papr-memory-key-1",
      "alg": "RS256",
      "n": "...",
      "e": "AQAB"
    }
  ]
}
```

### Test 3: Test GraphQL Proxy

```bash
# Test GraphQL endpoint
curl -X POST http://localhost:8000/v1/graphql \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query { __typename }"
  }'
```

### Test 4: Test with Real Query

```bash
curl -X POST http://localhost:8000/v1/graphql \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query GetTasks { tasks(options: { limit: 5 }) { id title status } }",
    "variables": {}
  }'
```

---

## Step 7: Configure Neo4j GraphQL Schema

Update your GraphQL schema with authentication configuration:

```graphql
extend schema
  @authentication(
    type: JWT
    jwks: {
      url: "https://memory.papr.ai/.well-known/jwks.json"
    }
  )

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
  status: String!
  user_id: String!
  workspace_id: String
  # ... other fields
}
```

Upload this schema to Neo4j using their admin API or console.

---

## Summary

That's it! The implementation is actually quite simple:

1. **JWT Service** (`services/jwt_service.py`) - Converts your existing auth → JWT
2. **JWKS Endpoint** (`/.well-known/jwks.json`) - Provides public key for Neo4j
3. **GraphQL Proxy** (`/v1/graphql`) - Forwards requests with JWT

Your existing authentication stays the same - clients still use API keys. The JWT is just an internal translation for Neo4j.

---

## Next Steps

- [ ] Generate RSA key pair
- [ ] Implement JWT service
- [ ] Create JWKS endpoint
- [ ] Create GraphQL proxy route
- [ ] Test locally
- [ ] Update Neo4j schema with @authorization directives
- [ ] Deploy to staging
- [ ] Test multi-tenant isolation
- [ ] Deploy to production

---

## Troubleshooting

### Issue: "Invalid JWT signature"
- **Cause**: Neo4j can't validate your JWT
- **Solution**: Check JWKS endpoint is accessible publicly, verify key format

### Issue: "Forbidden" errors even with valid auth
- **Cause**: @authorization directive not matching JWT claims
- **Solution**: Verify JWT has `user_id` claim, check @authorization WHERE clause

### Issue: "JWT expired"
- **Cause**: Token expiration time too short
- **Solution**: Increase `expires_in_minutes` in JWT generation

### Issue: JWKS endpoint returns 404
- **Cause**: Route not registered or CORS blocking Neo4j
- **Solution**: Check route is registered, add CORS headers to JWKS response
