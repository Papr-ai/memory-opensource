# GraphQL Quickstart Guide

Get started with PAPR Memory's GraphQL API in 5 minutes.

---

## Step 1: Get Your API Key

If you don't have an API key yet:

```bash
# Sign up at dashboard.papr.ai
# Navigate to Settings → API Keys
# Copy your API key
```

---

## Step 2: Test the GraphQL Endpoint

### Using cURL

```bash
curl -X POST https://memory.papr.ai/v1/graphql \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query { __typename }"
  }'
```

### Using the GraphQL Playground

Open your browser and navigate to:

```
http://localhost:8000/v1/graphql
```

(Development only - enter your API key when prompted)

---

## Step 3: Run Your First Query

### Get Tasks

```graphql
query GetTasks {
  tasks(options: { limit: 10 }) {
    id
    title
    status
    description
    createdAt
  }
}
```

### Get Projects with Tasks

```graphql
query GetProjectTasks($projectId: ID!) {
  project(id: $projectId) {
    id
    name
    description
    createdAt
    tasks {
      id
      title
      status
    }
  }
}
```

Variables:
```json
{
  "projectId": "your_project_id"
}
```

### Search Memories

```graphql
query SearchMemories($searchTerm: String!) {
  memories(
    where: { content_CONTAINS: $searchTerm }
    options: { limit: 20, sort: [{ createdAt: DESC }] }
  ) {
    id
    content
    topics
    createdAt
  }
}
```

Variables:
```json
{
  "searchTerm": "authentication"
}
```

---

## Step 4: Use in Your Application

### Python (with HTTP client)

```python
import httpx

GRAPHQL_URL = "https://memory.papr.ai/v1/graphql"
API_KEY = "your_api_key"

async def query_graphql(query: str, variables: dict = None):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            GRAPHQL_URL,
            json={
                "query": query,
                "variables": variables or {}
            },
            headers={
                "X-API-Key": API_KEY,
                "Content-Type": "application/json"
            }
        )
        return response.json()

# Usage
query = """
    query GetTasks {
        tasks(options: { limit: 5 }) {
            id
            title
            status
        }
    }
"""

result = await query_graphql(query)
print(result["data"]["tasks"])
```

### TypeScript (with fetch)

```typescript
const GRAPHQL_URL = 'https://memory.papr.ai/v1/graphql';
const API_KEY = 'your_api_key';

async function queryGraphQL(query: string, variables: any = {}) {
  const response = await fetch(GRAPHQL_URL, {
    method: 'POST',
    headers: {
      'X-API-Key': API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      variables,
    }),
  });

  return response.json();
}

// Usage
const query = `
  query GetTasks {
    tasks(options: { limit: 5 }) {
      id
      title
      status
    }
  }
`;

const result = await queryGraphQL(query);
console.log(result.data.tasks);
```

### JavaScript (with Apollo Client)

```javascript
import { ApolloClient, InMemoryCache, gql, HttpLink } from '@apollo/client';

const client = new ApolloClient({
  link: new HttpLink({
    uri: 'https://memory.papr.ai/v1/graphql',
    headers: {
      'X-API-Key': 'your_api_key',
    },
  }),
  cache: new InMemoryCache(),
});

// Query
const GET_TASKS = gql`
  query GetTasks {
    tasks(options: { limit: 5 }) {
      id
      title
      status
    }
  }
`;

const { data } = await client.query({ query: GET_TASKS });
console.log(data.tasks);
```

---

## Common Queries

### 1. Get All Tasks for a Project

```graphql
query GetProjectTasks($projectId: ID!) {
  project(id: $projectId) {
    name
    tasks {
      title
      status
      assignedTo {
        name
      }
    }
  }
}
```

### 2. Get Memories Related to a Task

```graphql
query GetTaskMemories($taskId: ID!) {
  task(id: $taskId) {
    title
    memories: relatedMemories {
      content
      createdAt
    }
  }
}
```

### 3. Search by Multiple Criteria

```graphql
query SearchMemories($topic: String!, $afterDate: String!) {
  memories(
    where: {
      topics_CONTAINS: $topic
      createdAt_GT: $afterDate
    }
    options: { limit: 10 }
  ) {
    content
    topics
    createdAt
  }
}
```

---

## GraphQL Features

### Automatic Filtering

All queries are automatically filtered by your `user_id` and `workspace_id`. You can only access your own data - no need to add filters manually!

### Type Safety

GraphQL provides strong typing. Your IDE will autocomplete field names and catch errors before runtime.

### Efficient Data Fetching

Request exactly the fields you need - no over-fetching or under-fetching.

### Relationship Traversal

Navigate relationships in a single query:

```graphql
query {
  project(id: "proj_123") {
    tasks {
      assignedTo {
        profile {
          email
        }
      }
    }
  }
}
```

---

## Next Steps

- **Explore the Schema**: Use GraphQL Playground's "Docs" tab
- **Read the Full Architecture**: See `docs/roadmap/graphql/graphql-architecture.md`
- **Check SDK Examples**: See Python and TypeScript SDK documentation
- **Learn Advanced Queries**: Filtering, sorting, pagination

---

## Troubleshooting

### "Missing authentication"
- Make sure you're sending the `X-API-Key` header
- Verify your API key is valid

### "Forbidden" error
- The resource doesn't belong to your user_id/workspace_id
- Check that you're querying the correct IDs

### Slow queries
- Use `limit` to restrict result size
- Avoid deeply nested queries (max depth: 10)

---

## Support

- **Documentation**: https://docs.papr.ai
- **Issues**: https://github.com/papr/memory/issues
- **Email**: support@papr.ai
