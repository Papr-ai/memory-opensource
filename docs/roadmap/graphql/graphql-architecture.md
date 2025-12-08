# GraphQL Architecture for PAPR Memory

**Status**: Architecture Design
**Last Updated**: 2025-10-30
**Owner**: PAPR Engineering Team

---

## Table of Contents
- [Executive Summary](#executive-summary)
- [Architecture Overview](#architecture-overview)
- [Neo4j GraphQL Integration](#neo4j-graphql-integration)
- [Multi-Tenant Authorization](#multi-tenant-authorization)
- [FastAPI Proxy Layer](#fastapi-proxy-layer)
- [Python SDK Integration](#python-sdk-integration)
- [TypeScript SDK Integration](#typescript-sdk-integration)
- [Schema Management](#schema-management)
- [Implementation Roadmap](#implementation-roadmap)
- [Security Model](#security-model)
- [Developer Experience](#developer-experience)

---

## Executive Summary

This document outlines the technical architecture for enabling GraphQL queries on PAPR Memory, a multi-tenant knowledge graph system. The solution leverages **Neo4j's hosted GraphQL endpoint** with JWT-based authorization to provide developers with type-safe, efficient GraphQL access to their memory data.

### Key Design Decisions

1. **Use Neo4j's Hosted GraphQL Endpoint**
   - Endpoint: `https://de7df98e-graphql.production-orch-0042.neo4j.io/graphql`
   - Rationale: Leverage Neo4j's native GraphQL support instead of building custom resolvers
   - Benefits: Auto-generated queries, optimized graph traversal, built-in authorization directives

2. **JWT-Based Multi-Tenant Authorization**
   - Use Neo4j's `@authorization` directive to filter data by `user_id` and `workspace_id`
   - FastAPI proxy generates JWT tokens from API keys
   - Neo4j validates JWT and automatically applies tenant isolation

3. **Dual SDK Approach**
   - **Python SDK**: HTTP client with GraphQL query builder (or Strawberry client)
   - **TypeScript SDK**: Apollo Client with generated types
   - Both SDKs use the FastAPI proxy for authentication

4. **Schema Management**
   - Default schema: ~1665 types from existing Neo4j graph
   - Custom schemas: User-defined types merged dynamically
   - Schema introspection available via GraphQL endpoint

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                       │
│              (Python SDK / TypeScript SDK)                   │
└────────────┬────────────────────────────────────────────────┘
             │
             │ GraphQL Query + API Key
             │
┌────────────▼────────────────────────────────────────────────┐
│              FastAPI Proxy Layer                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │     Authentication & JWT Generation                   │  │
│  │  - Validate API key / Bearer token / Session token   │  │
│  │  - Resolve user_id, workspace_id, roles             │  │
│  │  - Generate JWT with claims                          │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │     Request Forwarding                               │  │
│  │  - Add Authorization: Bearer <JWT> header           │  │
│  │  - Forward GraphQL query to Neo4j endpoint          │  │
│  │  - Return response to client                        │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────┬────────────────────────────────────────────────┘
             │
             │ GraphQL Query + JWT
             │
┌────────────▼────────────────────────────────────────────────┐
│         Neo4j Hosted GraphQL Endpoint                        │
│  https://de7df98e-graphql.production-orch-0042.neo4j.io/... │
│  ┌──────────────────────────────────────────────────────┐  │
│  │     JWT Validation                                    │  │
│  │  - Verify JWT signature                              │  │
│  │  - Extract claims (user_id, workspace_id, roles)    │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │     @authorization Directive Application             │  │
│  │  - Apply WHERE filters from @authorization          │  │
│  │  - Filter by user_id, workspace_id automatically    │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │     Query Execution                                   │  │
│  │  - Execute Cypher query with filters                │  │
│  │  - Traverse graph efficiently                        │  │
│  │  - Return results                                    │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│              Neo4j Graph Database                            │
│           (Multi-Tenant Data Store)                          │
└──────────────────────────────────────────────────────────────┘
```

---

## Neo4j GraphQL Integration

### How Neo4j GraphQL Works

Neo4j provides a **hosted GraphQL endpoint** that automatically generates GraphQL queries from your graph schema. It uses GraphQL SDL (Schema Definition Language) with special directives for authorization.

### Schema Definition with Authorization

Your existing schema at `/Users/shawkatkabbara/Documents/GitHub/memory/models/papr_graphql_default_schema.py` needs to be updated with Neo4j's authorization directives:

```graphql
extend schema
  @link(url: "https://specs.apollo.dev/federation/v2.6", import: ["@key"])

# Authentication configuration for JWT
extend schema
  @authentication(
    type: JWT
    jwks: {
      url: "https://memory.papr.ai/.well-known/jwks.json"
    }
  )

# Example: Task type with authorization
type Task @node
  @authorization(
    validate: [
      {
        when: [BEFORE]
        operations: [READ, UPDATE, DELETE, CREATE]
        where: {
          node: {
            user_id: "$jwt.user_id"
          }
        }
      }
      {
        when: [BEFORE]
        operations: [READ, UPDATE, DELETE, CREATE]
        where: {
          OR: [
            { node: { workspace_id: "$jwt.workspace_id" } }
            { node: { workspace_id: null } }
          ]
        }
      }
    ]
  )
{
  id: String
  title: String!
  status: String!
  description: String
  user_id: String!
  workspace_id: String
  createdAt: String!

  # Relationships
  project: Project @relationship(type: "BELONGS_TO", direction: OUT)
  assignedTo: Person @relationship(type: "ASSIGNED_TO", direction: OUT)
}

# Example: Project type with authorization
type Project @node
  @authorization(
    validate: [
      {
        when: [BEFORE]
        operations: [READ, UPDATE, DELETE, CREATE]
        where: { node: { user_id: "$jwt.user_id" } }
      }
      {
        when: [BEFORE]
        operations: [READ, UPDATE, DELETE, CREATE]
        where: {
          OR: [
            { node: { workspace_id: "$jwt.workspace_id" } }
            { node: { workspace_id: null } }
          ]
        }
      }
    ]
  )
{
  id: String!
  name: String!
  description: String
  user_id: String!
  workspace_id: String
  createdAt: String!

  # Relationships (automatically filtered)
  tasks: [Task!]! @relationship(type: "HAS", direction: OUT)
  members: [Person!]! @relationship(type: "MEMBER_OF", direction: IN)
}
```

### Key Authorization Concepts

#### 1. `@authentication` Directive

Tells Neo4j GraphQL how to validate JWT tokens:

```graphql
extend schema
  @authentication(
    type: JWT
    jwks: {
      url: "https://memory.papr.ai/.well-known/jwks.json"
    }
  )
```

**Requirements**:
- You need to create a JWKS (JSON Web Key Set) endpoint at `/.well-known/jwks.json`
- This endpoint returns your public keys for JWT verification
- Neo4j will fetch this and cache it

#### 2. `@authorization` Directive

Defines row-level security rules:

```graphql
@authorization(
  validate: [
    {
      when: [BEFORE]  # Apply filter BEFORE query execution
      operations: [READ, UPDATE, DELETE, CREATE]
      where: {
        node: { user_id: "$jwt.user_id" }  # Reference JWT claims
      }
    }
  ]
)
```

**How It Works**:
- `$jwt.user_id` references the `user_id` claim in the JWT
- Neo4j automatically injects `WHERE node.user_id = <jwt_user_id>` into all queries
- This happens at the Cypher query level, before execution
- **No way to bypass** - it's enforced by Neo4j GraphQL

#### 3. Multiple Authorization Rules

For workspace + user filtering:

```graphql
@authorization(
  validate: [
    # Rule 1: Must match user_id
    {
      when: [BEFORE]
      operations: [READ, UPDATE, DELETE, CREATE]
      where: { node: { user_id: "$jwt.user_id" } }
    },
    # Rule 2: Must match workspace_id OR be null (personal)
    {
      when: [BEFORE]
      operations: [READ, UPDATE, DELETE, CREATE]
      where: {
        OR: [
          { node: { workspace_id: "$jwt.workspace_id" } }
          { node: { workspace_id: null } }
        ]
      }
    }
  ]
)
```

**Result**: Only returns data where:
- `user_id` matches the JWT claim **AND**
- `workspace_id` matches the JWT claim OR is null

---

## Multi-Tenant Authorization

### JWT Token Structure

The FastAPI proxy generates JWT tokens with the following claims:

```json
{
  "sub": "user_abc123",           // Subject (user ID)
  "user_id": "user_abc123",       // Custom claim
  "workspace_id": "ws_xyz789",    // Custom claim
  "end_user_id": "enduser_456",   // Optional
  "roles": ["developer", "admin"],
  "permissions": ["read", "write"],
  "iss": "https://memory.papr.ai",
  "aud": "neo4j-graphql",
  "exp": 1735564800,              // Expiration (Unix timestamp)
  "iat": 1735561200               // Issued at
}
```

### JWT Generation in FastAPI

```python
# /Users/shawkatkabbara/Documents/GitHub/memory/services/jwt_service.py

from datetime import datetime, timedelta, UTC
import jwt
from typing import Optional

class JWTService:
    """Generate JWTs for Neo4j GraphQL authorization"""

    def __init__(self):
        # Load your private key for signing
        self.private_key = self._load_private_key()
        self.algorithm = "RS256"  # Use RSA signing
        self.issuer = "https://memory.papr.ai"
        self.audience = "neo4j-graphql"

    def _load_private_key(self) -> str:
        """Load private key from environment or file"""
        import os
        private_key_path = os.getenv("JWT_PRIVATE_KEY_PATH")
        with open(private_key_path, 'r') as f:
            return f.read()

    def generate_token(
        self,
        user_id: str,
        workspace_id: Optional[str] = None,
        end_user_id: Optional[str] = None,
        roles: list[str] = None,
        permissions: list[str] = None,
        expires_in_minutes: int = 60
    ) -> str:
        """
        Generate a JWT token for Neo4j GraphQL.

        Args:
            user_id: The user's unique identifier
            workspace_id: The workspace identifier (optional)
            end_user_id: End user identifier (optional)
            roles: List of user roles
            permissions: List of user permissions
            expires_in_minutes: Token expiration time in minutes

        Returns:
            Signed JWT token string
        """
        now = datetime.now(UTC)
        expiration = now + timedelta(minutes=expires_in_minutes)

        payload = {
            "sub": user_id,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "end_user_id": end_user_id,
            "roles": roles or [],
            "permissions": permissions or [],
            "iss": self.issuer,
            "aud": self.audience,
            "exp": int(expiration.timestamp()),
            "iat": int(now.timestamp())
        }

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        # Sign the token
        token = jwt.encode(
            payload,
            self.private_key,
            algorithm=self.algorithm
        )

        return token
```

### JWKS Endpoint

Neo4j needs to fetch your public key to validate JWTs:

```python
# /Users/shawkatkabbara/Documents/GitHub/memory/routers/v1/jwks_routes.py

from fastapi import APIRouter, Response
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import json
import base64

router = APIRouter(tags=["JWKS"])

@router.get("/.well-known/jwks.json")
async def get_jwks():
    """
    JWKS endpoint for Neo4j GraphQL JWT validation.

    Returns the public keys used to verify JWT signatures.
    """
    # Load your public key
    public_key_path = os.getenv("JWT_PUBLIC_KEY_PATH")
    with open(public_key_path, 'rb') as f:
        public_key = serialization.load_pem_public_key(
            f.read(),
            backend=default_backend()
        )

    # Convert to JWK format
    public_numbers = public_key.public_numbers()

    def encode_int(n):
        """Encode integer as base64url"""
        # Convert to bytes, then base64url encode
        b = n.to_bytes((n.bit_length() + 7) // 8, byteorder='big')
        return base64.urlsafe_b64encode(b).rstrip(b'=').decode('utf-8')

    jwk = {
        "kty": "RSA",
        "use": "sig",
        "kid": "papr-memory-key-1",  # Key ID
        "alg": "RS256",
        "n": encode_int(public_numbers.n),  # Modulus
        "e": encode_int(public_numbers.e),  # Exponent
    }

    jwks = {
        "keys": [jwk]
    }

    return Response(
        content=json.dumps(jwks),
        media_type="application/json",
        headers={
            "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
        }
    )
```

---

## FastAPI Proxy Layer

### GraphQL Proxy Endpoint

```python
# /Users/shawkatkabbara/Documents/GitHub/memory/routers/v1/graphql_routes.py

from fastapi import APIRouter, Request, Depends, Response, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
from typing import Optional, Dict, Any
import httpx
from services.auth_utils import get_user_from_token_optimized
from services.jwt_service import JWTService
from memory.memory_graph import MemoryGraph
from services.utils import get_memory_graph
import time

router = APIRouter(prefix="/graphql", tags=["GraphQL"])

# Security schemes
bearer_auth = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
session_token_header = APIKeyHeader(name="X-Session-Token", auto_error=False)

# Neo4j GraphQL endpoint
NEO4J_GRAPHQL_ENDPOINT = "https://de7df98e-graphql.production-orch-0042.neo4j.io/graphql"

# JWT service
jwt_service = JWTService()

@router.post("")
async def graphql_proxy(
    request: Request,
    response: Response,
    api_key: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Depends(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
):
    """
    GraphQL proxy endpoint that:
    1. Authenticates the request using existing auth system
    2. Generates a JWT token with user claims
    3. Forwards the GraphQL query to Neo4j with JWT
    4. Returns the response
    """

    # --- Authentication (reuse existing system) ---
    client_type = request.headers.get('X-Client-Type', 'graphql_client')

    try:
        async with httpx.AsyncClient() as httpx_client:
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
        raise HTTPException(401, f"Authentication failed: {str(e)}")

    if not auth_response:
        raise HTTPException(401, "Invalid authentication")

    # Extract user information
    user_id = auth_response.developer_id
    workspace_id = auth_response.workspace_id
    end_user_id = auth_response.end_user_id

    # --- Generate JWT for Neo4j GraphQL ---
    neo4j_jwt = jwt_service.generate_token(
        user_id=user_id,
        workspace_id=workspace_id,
        end_user_id=end_user_id,
        roles=["developer"],  # Could come from auth_response
        permissions=["read", "write"],  # Could come from ACL
        expires_in_minutes=60
    )

    # --- Parse GraphQL request body ---
    try:
        body = await request.json()
        query = body.get("query")
        variables = body.get("variables", {})
        operation_name = body.get("operationName")
    except Exception as e:
        raise HTTPException(400, f"Invalid GraphQL request: {str(e)}")

    # --- Forward to Neo4j GraphQL with JWT ---
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

            # Return Neo4j's response
            return Response(
                content=neo4j_response.content,
                status_code=neo4j_response.status_code,
                media_type="application/json"
            )

        except httpx.TimeoutException:
            raise HTTPException(504, "Neo4j GraphQL request timeout")
        except Exception as e:
            raise HTTPException(500, f"Neo4j GraphQL error: {str(e)}")


@router.get("")
async def graphql_playground():
    """
    Serve GraphQL Playground for development.
    Disable in production.
    """
    import os
    if os.getenv("ENVIRONMENT") == "production":
        raise HTTPException(404, "GraphQL Playground disabled in production")

    # Serve GraphQL Playground HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>PAPR Memory GraphQL</title>
        <link rel="stylesheet" href="https://unpkg.com/graphiql/graphiql.min.css" />
    </head>
    <body style="margin: 0;">
        <div id="graphiql" style="height: 100vh;"></div>

        <script
            crossorigin
            src="https://unpkg.com/react/umd/react.production.min.js"
        ></script>
        <script
            crossorigin
            src="https://unpkg.com/react-dom/umd/react-dom.production.min.js"
        ></script>
        <script
            crossorigin
            src="https://unpkg.com/graphiql/graphiql.min.js"
        ></script>

        <script>
            const fetcher = GraphiQL.createFetcher({{
                url: '/v1/graphql',
                headers: {{
                    'X-API-Key': 'YOUR_API_KEY_HERE'
                }}
            }});

            ReactDOM.render(
                React.createElement(GraphiQL, {{ fetcher: fetcher }}),
                document.getElementById('graphiql'),
            );
        </script>
    </body>
    </html>
    """

    return Response(content=html, media_type="text/html")


@router.get("/schema")
async def get_graphql_schema(
    api_key: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
):
    """
    Get the GraphQL schema SDL (Schema Definition Language).

    This introspects the Neo4j GraphQL endpoint and returns the schema.
    """

    # Authenticate first
    client_type = request.headers.get('X-Client-Type', 'graphql_client')

    async with httpx.AsyncClient() as httpx_client:
        if api_key:
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
        else:
            raise HTTPException(401, "Missing authentication")

    if not auth_response:
        raise HTTPException(401, "Invalid authentication")

    # Generate JWT
    neo4j_jwt = jwt_service.generate_token(
        user_id=auth_response.developer_id,
        workspace_id=auth_response.workspace_id
    )

    # Introspection query
    introspection_query = """
    query IntrospectionQuery {
        __schema {
            queryType { name }
            mutationType { name }
            subscriptionType { name }
            types {
                ...FullType
            }
            directives {
                name
                description
                locations
                args {
                    ...InputValue
                }
            }
        }
    }

    fragment FullType on __Type {
        kind
        name
        description
        fields(includeDeprecated: true) {
            name
            description
            args {
                ...InputValue
            }
            type {
                ...TypeRef
            }
            isDeprecated
            deprecationReason
        }
        inputFields {
            ...InputValue
        }
        interfaces {
            ...TypeRef
        }
        enumValues(includeDeprecated: true) {
            name
            description
            isDeprecated
            deprecationReason
        }
        possibleTypes {
            ...TypeRef
        }
    }

    fragment InputValue on __InputValue {
        name
        description
        type { ...TypeRef }
        defaultValue
    }

    fragment TypeRef on __Type {
        kind
        name
        ofType {
            kind
            name
            ofType {
                kind
                name
                ofType {
                    kind
                    name
                    ofType {
                        kind
                        name
                    }
                }
            }
        }
    }
    """

    async with httpx.AsyncClient() as client:
        neo4j_response = await client.post(
            NEO4J_GRAPHQL_ENDPOINT,
            json={"query": introspection_query},
            headers={
                "Authorization": f"Bearer {neo4j_jwt}",
                "Content-Type": "application/json"
            }
        )

        return neo4j_response.json()
```

---

## Python SDK Integration

### Option 1: Simple HTTP Client (Recommended for Start)

```python
# /Users/shawkatkabbara/Documents/GitHub/papr-PythonSDK/src/papr_memory/graphql_client.py

from typing import Optional, Dict, Any, TypeVar, Generic
import httpx
from pydantic import BaseModel

T = TypeVar('T')

class GraphQLResponse(BaseModel, Generic[T]):
    """GraphQL response wrapper"""
    data: Optional[T] = None
    errors: Optional[list[Dict[str, Any]]] = None

class PaprGraphQLClient:
    """
    PAPR Memory GraphQL Client for Python.

    Usage:
        client = PaprGraphQLClient(api_key="your_api_key")

        query = '''
            query GetProjectTasks($projectId: ID!) {
                project(id: $projectId) {
                    name
                    tasks { title status }
                }
            }
        '''

        result = await client.execute(query, {"projectId": "proj_123"})
        print(result.data)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://memory.papr.ai",
        timeout: float = 30.0
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.graphql_url = f"{self.base_url}/v1/graphql"
        self.timeout = timeout

    async def execute(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None
    ) -> GraphQLResponse:
        """
        Execute a GraphQL query.

        Args:
            query: GraphQL query string
            variables: Query variables
            operation_name: Operation name (optional)

        Returns:
            GraphQLResponse with data or errors
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.graphql_url,
                json={
                    "query": query,
                    "variables": variables or {},
                    "operationName": operation_name
                },
                headers={
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json"
                },
                timeout=self.timeout
            )

            response.raise_for_status()

            result = response.json()
            return GraphQLResponse(**result)

    async def query(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a query and return data directly.
        Raises exception if errors are present.
        """
        result = await self.execute(query, variables)

        if result.errors:
            error_messages = [e.get('message', str(e)) for e in result.errors]
            raise Exception(f"GraphQL errors: {', '.join(error_messages)}")

        return result.data


# Example usage
async def example_usage():
    client = PaprGraphQLClient(api_key="your_api_key")

    # Query projects with tasks
    query = """
        query GetProjectTasks($projectId: ID!) {
            project(id: $projectId) {
                id
                name
                createdAt
                tasks {
                    id
                    title
                    status
                    description
                }
            }
        }
    """

    result = await client.query(query, {"projectId": "proj_123"})

    project = result["project"]
    print(f"Project: {project['name']}")
    for task in project["tasks"]:
        print(f"  - {task['title']}: {task['status']}")
```

### Option 2: Type-Safe Client with Code Generation

Generate Python types from GraphQL schema:

```bash
# Install GraphQL code generator
pip install ariadne-codegen

# Generate types
ariadne-codegen client \
  --schema-url https://memory.papr.ai/v1/graphql/schema \
  --header "X-API-Key: your_key" \
  --output-dir src/papr_memory/generated
```

Then use generated types:

```python
from papr_memory.generated import (
    GetProjectTasksQuery,
    GetProjectTasksQueryVariables
)
from papr_memory import PaprGraphQLClient

client = PaprGraphQLClient(api_key="your_key")

# Type-safe query
variables = GetProjectTasksQueryVariables(projectId="proj_123")
result: GetProjectTasksQuery = await client.query(
    GetProjectTasksQuery.query,
    variables.dict()
)

# Full autocomplete and type checking
project = result.project
if project:
    print(project.name)  # ✅ Type-safe
    for task in project.tasks:
        print(task.title)  # ✅ Autocomplete works
```

---

## TypeScript SDK Integration

### Apollo Client Setup

```typescript
// /Users/shawkatkabbara/Documents/GitHub/papr-typescript-sdk/src/graphql/client.ts

import {
  ApolloClient,
  InMemoryCache,
  HttpLink,
  ApolloLink,
  from
} from '@apollo/client';

export interface PaprGraphQLClientConfig {
  apiKey?: string;
  baseUrl?: string;
}

export class PaprGraphQLClient {
  private client: ApolloClient<any>;

  constructor(config: PaprGraphQLClientConfig) {
    const { apiKey, baseUrl = 'https://memory.papr.ai' } = config;

    // HTTP link to PAPR Memory GraphQL endpoint
    const httpLink = new HttpLink({
      uri: `${baseUrl}/v1/graphql`,
    });

    // Auth middleware to add API key
    const authLink = new ApolloLink((operation, forward) => {
      operation.setContext({
        headers: {
          'X-API-Key': apiKey || '',
          'Content-Type': 'application/json',
        },
      });
      return forward(operation);
    });

    // Create Apollo Client
    this.client = new ApolloClient({
      link: from([authLink, httpLink]),
      cache: new InMemoryCache(),
      defaultOptions: {
        watchQuery: {
          fetchPolicy: 'network-only',
        },
        query: {
          fetchPolicy: 'network-only',
        },
      },
    });
  }

  /**
   * Get the underlying Apollo Client instance
   */
  getClient(): ApolloClient<any> {
    return this.client;
  }
}

// Example usage
const client = new PaprGraphQLClient({
  apiKey: 'your_api_key',
});

export default client;
```

### Type Generation with GraphQL Codegen

```bash
# Install dependencies
npm install --save-dev @graphql-codegen/cli @graphql-codegen/typescript @graphql-codegen/typescript-operations @graphql-codegen/typescript-react-apollo

# Create codegen.yml
```

```yaml
# codegen.yml
schema:
  - https://memory.papr.ai/v1/graphql/schema:
      headers:
        X-API-Key: ${PAPR_API_KEY}

documents:
  - 'src/**/*.graphql'
  - 'src/**/*.tsx'

generates:
  src/generated/graphql.ts:
    plugins:
      - typescript
      - typescript-operations
      - typescript-react-apollo
    config:
      withHooks: true
      withComponent: false
      withHOC: false
```

### Type-Safe Queries

```typescript
// src/queries/projects.graphql
query GetProjectTasks($projectId: ID!) {
  project(id: $projectId) {
    id
    name
    createdAt
    tasks {
      id
      title
      status
      description
      assignedTo {
        id
        name
      }
    }
  }
}

mutation CreateTask($input: CreateTaskInput!) {
  createTask(input: $input) {
    id
    title
    status
  }
}
```

```typescript
// src/components/ProjectView.tsx
import { useGetProjectTasksQuery } from '../generated/graphql';

export function ProjectView({ projectId }: { projectId: string }) {
  const { data, loading, error } = useGetProjectTasksQuery({
    variables: { projectId },
  });

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  const project = data?.project;

  return (
    <div>
      <h1>{project?.name}</h1>
      <ul>
        {project?.tasks.map(task => (
          <li key={task.id}>
            {task.title} - {task.status}
            {task.assignedTo && ` (${task.assignedTo.name})`}
          </li>
        ))}
      </ul>
    </div>
  );
}
```

---

## Schema Management

### Default Schema Loading

Your existing schema at `/Users/shawkatkabbara/Documents/GitHub/memory/models/papr_graphql_default_schema.py` needs to be:

1. **Updated with @authorization directives**
2. **Uploaded to Neo4j**

```python
# Script to update Neo4j GraphQL schema
import httpx
import os

async def update_neo4j_schema():
    """Update the Neo4j GraphQL schema with authorization directives"""

    # Read your schema file
    schema_path = "/Users/shawkatkabbara/Documents/GitHub/memory/models/papr_graphql_default_schema.py"
    with open(schema_path, 'r') as f:
        schema_sdl = f.read()

    # Neo4j GraphQL schema endpoint
    neo4j_admin_endpoint = "https://de7df98e-graphql.production-orch-0042.neo4j.io/admin"
    neo4j_admin_password = os.getenv("NEO4J_ADMIN_PASSWORD")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            neo4j_admin_endpoint,
            json={
                "query": """
                    mutation UpdateSchema($schema: String!) {
                        updateSchema(schema: $schema)
                    }
                """,
                "variables": {
                    "schema": schema_sdl
                }
            },
            auth=("admin", neo4j_admin_password)
        )

        print(response.json())

# Run this when you update your schema
asyncio.run(update_neo4j_schema())
```

### Custom User Schemas

When users create custom schemas via `/v1/schemas`:

```python
# Merge custom schema with default schema
def merge_user_schema(default_schema: str, user_schema: UserGraphSchema) -> str:
    """
    Merge user's custom schema with default schema.

    1. Parse user schema
    2. Add @authorization directives
    3. Append to default schema
    """

    custom_types = []

    for node_type_name, node_type in user_schema.node_types.items():
        # Generate GraphQL type with authorization
        type_def = f"""
type {node_type_name} @node
  @authorization(
    validate: [
      {{
        when: [BEFORE]
        operations: [READ, UPDATE, DELETE, CREATE]
        where: {{ node: {{ user_id: "$jwt.user_id" }} }}
      }}
    ]
  )
{{
  id: ID!
  user_id: String!
  workspace_id: String
"""

        # Add properties
        for prop_name, prop_def in node_type.properties.items():
            graphql_type = map_to_graphql_type(prop_def['type'])
            required = "!" if prop_def.get('required') else ""
            type_def += f"  {prop_name}: {graphql_type}{required}\n"

        type_def += "}\n"
        custom_types.append(type_def)

    # Combine with default schema
    merged_schema = default_schema + "\n\n" + "\n\n".join(custom_types)

    return merged_schema
```

---

## Implementation Roadmap

### Phase 1: JWT Infrastructure (Week 1)

**Tasks**:
1. Create JWT service (`services/jwt_service.py`)
2. Generate RSA key pair for signing
3. Implement JWKS endpoint (`/.well-known/jwks.json`)
4. Test JWT generation and validation

**Deliverable**: Working JWT generation from API keys

### Phase 2: Schema Authorization (Week 2)

**Tasks**:
1. Update default schema with `@authorization` directives
2. Add `@authentication` directive to schema
3. Upload schema to Neo4j GraphQL endpoint
4. Test authorization filters with sample queries

**Deliverable**: Neo4j GraphQL enforcing multi-tenant isolation

### Phase 3: FastAPI Proxy (Week 3)

**Tasks**:
1. Create GraphQL proxy endpoint (`/v1/graphql`)
2. Implement auth → JWT flow
3. Add request/response logging
4. Set up GraphQL Playground for development

**Deliverable**: Working proxy endpoint

### Phase 4: Python SDK (Week 4)

**Tasks**:
1. Create `PaprGraphQLClient` class
2. Add query/mutation methods
3. Implement type generation (optional)
4. Write SDK documentation
5. Add usage examples

**Deliverable**: Python SDK with GraphQL support

### Phase 5: TypeScript SDK (Week 5)

**Tasks**:
1. Set up Apollo Client wrapper
2. Configure GraphQL Code Generator
3. Create React hooks
4. Write SDK documentation
5. Add usage examples

**Deliverable**: TypeScript SDK with GraphQL support

### Phase 6: Custom Schema Support (Week 6)

**Tasks**:
1. Implement schema merging logic
2. Update `/v1/schemas` endpoint to trigger GraphQL schema update
3. Add schema validation
4. Test custom types with authorization

**Deliverable**: Custom schemas work with GraphQL

### Phase 7: Production Hardening (Week 7)

**Tasks**:
1. Add rate limiting
2. Implement query complexity analysis
3. Set up monitoring and alerting
4. Security audit
5. Performance optimization

**Deliverable**: Production-ready GraphQL API

---

## Security Model

### Authorization Flow

```
1. Client → FastAPI: GraphQL query + API key
2. FastAPI: Validate API key
3. FastAPI: Extract user_id, workspace_id
4. FastAPI: Generate JWT with claims
5. FastAPI → Neo4j: GraphQL query + JWT
6. Neo4j: Validate JWT signature (JWKS)
7. Neo4j: Extract claims from JWT
8. Neo4j: Apply @authorization filters
9. Neo4j: Execute Cypher with WHERE clauses
10. Neo4j → FastAPI: Results
11. FastAPI → Client: Results
```

### Security Guarantees

✅ **No Cross-Tenant Data Access**
- `@authorization` directive enforces `user_id` and `workspace_id` filters
- Filters applied at Cypher query level (before execution)
- No way to bypass via GraphQL query

✅ **JWT Validation**
- Neo4j validates JWT signature using JWKS
- Expired tokens are rejected
- Invalid signatures are rejected

✅ **Rate Limiting**
- Per-organization rate limits
- Per-user rate limits
- Query complexity limits

✅ **Audit Logging**
- All GraphQL queries logged
- Failed auth attempts logged
- Unusual query patterns flagged

### Potential Security Concerns

⚠️ **GraphQL Introspection**
- Introspection reveals schema to all authenticated users
- Consider disabling in production or adding custom authorization
- Solution: Disable introspection or return user-specific schema

⚠️ **Query Depth Attacks**
- Deeply nested queries can overload database
- Solution: Add query depth limiting in FastAPI proxy

⚠️ **JWT Secret Leakage**
- If private key is compromised, attacker can generate valid JWTs
- Solution: Rotate keys regularly, store securely (AWS Secrets Manager, Vault)

---

## Developer Experience

### GraphQL Playground

Developers can test queries interactively:

```
GET https://memory.papr.ai/v1/graphql

(Opens GraphQL Playground with schema documentation)
```

### Sample Queries

**Get all tasks for a project:**
```graphql
query GetProjectTasks($projectId: ID!) {
  project(id: $projectId) {
    id
    name
    tasks {
      id
      title
      status
      assignedTo {
        name
        email
      }
    }
  }
}
```

**Search memories:**
```graphql
query SearchMemories($query: String!, $limit: Int = 10) {
  memories(
    where: { content_CONTAINS: $query }
    options: { limit: $limit, sort: [{ createdAt: DESC }] }
  ) {
    id
    content
    topics
    createdAt
  }
}
```

**Create a task:**
```graphql
mutation CreateTask($input: TaskCreateInput!) {
  createTasks(input: [$input]) {
    tasks {
      id
      title
      status
    }
  }
}
```

### Error Handling

GraphQL errors are returned in standard format:

```json
{
  "errors": [
    {
      "message": "Forbidden",
      "extensions": {
        "code": "FORBIDDEN",
        "exception": {
          "message": "User does not have permission to access this resource"
        }
      }
    }
  ],
  "data": null
}
```

### Documentation

Auto-generated documentation from schema:

- Field descriptions
- Required vs optional fields
- Relationship traversal
- Available filters and sorting options

---

## Next Steps

1. **Review this architecture** with the team
2. **Set up development environment**:
   - Generate RSA key pair for JWT signing
   - Configure JWKS endpoint
   - Test JWT generation
3. **Start Phase 1**: JWT infrastructure
4. **Create detailed task breakdown** in PAPR Memory

---

## Appendix: Neo4j GraphQL Resources

- [Neo4j GraphQL Docs](https://neo4j.com/docs/graphql/)
- [Authorization Directives](https://neo4j.com/docs/graphql/7/directives/)
- [Authentication Setup](https://neo4j.com/docs/graphql/7/authentication/)
- [Apollo Federation](https://neo4j.com/docs/graphql/7/integrations/apollo-federation/)
- [Relay Compatibility](https://neo4j.com/docs/graphql/7/integrations/relay-compatibility/)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-30
**Maintained By**: PAPR Engineering Team
