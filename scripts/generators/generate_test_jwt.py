#!/usr/bin/env python3
"""
Generate a test JWT token for Neo4j GraphQL testing
Reads JWT keys directly from files, no dependencies on .env
"""
import jwt
import time
from datetime import datetime, timedelta, UTC
from pathlib import Path

# Read private key from file
private_key_path = Path(__file__).parent / "keys" / "jwt-private.pem"
with open(private_key_path, 'r') as f:
    private_key = f.read()

# Generate JWT payload
now = datetime.now(UTC)
expiration = now + timedelta(minutes=60)

payload = {
    "sub": "test_user_123",
    "user_id": "test_user_123",
    "workspace_id": "test_workspace_456",
    "roles": ["developer"],
    "iss": "https://memory.papr.ai",
    "aud": "neo4j-graphql",
    "exp": int(expiration.timestamp()),
    "iat": int(now.timestamp())
}

# Sign token
token = jwt.encode(payload, private_key, algorithm="RS256")

print("✅ Generated JWT token:")
print(token)
print("\n" + "="*80)
print("Test with curl:")
print("="*80)
print(f"""
curl -v -X POST 'https://de7df98e-graphql.production-orch-0042.neo4j.io/graphql' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer {token}' \\
  -d '{{"query":"{{ __schema {{ queryType {{ name }} }} }}"}}'
""")
