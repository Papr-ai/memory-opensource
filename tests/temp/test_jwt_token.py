#!/usr/bin/env python3
"""
Generate a test JWT token for Neo4j GraphQL testing
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import find_dotenv, load_dotenv
from services.jwt_service import get_jwt_service

# Load environment
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

def main():
    try:
        # Generate a test JWT
        jwt_service = get_jwt_service()
        test_token = jwt_service.generate_token(
            user_id="test_user_123",
            workspace_id="test_workspace_456",
            roles=["developer"],
            expires_in_minutes=60
        )

        print("✅ Generated JWT token:")
        print(test_token)
        print("\n" + "="*80)
        print("Use this curl command to test Neo4j GraphQL:")
        print("="*80)
        print(f"""
curl -v -X POST "https://de7df98e-graphql.production-orch-0042.neo4j.io/graphql" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {test_token}" \\
  -d '{{"query":"{{ __schema {{ types {{ name }} }} }}"}}'
""")

        return 0

    except Exception as e:
        print(f"❌ Error generating JWT: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
