#!/usr/bin/env python3
"""
Test TensorLake API endpoints to find the correct ones
"""

import asyncio
import httpx
import os
from dotenv import load_dotenv

async def test_tensorlake_endpoints():
    """Test various TensorLake endpoints to find the correct ones"""

    # Load environment variables conditionally
    use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
    if use_dotenv:
        load_dotenv()

    api_key = os.getenv("TENSORLAKE_API_KEY")
    base_url = os.getenv("TENSORLAKE_BASE_URL", "https://api.tensorlake.ai")

    print(f"Testing TensorLake endpoints...")
    print(f"Base URL: {base_url}")
    print(f"API Key: {api_key[:20]}...")

    headers = {"Authorization": f"Bearer {api_key}"}

    # Test various endpoints
    test_endpoints = [
        "/",
        "/health",
        "/v1/health",
        "/v1/status",
        "/v1/process",
        "/api/health",
        "/api/v1/health",
        "/status"
    ]

    async with httpx.AsyncClient(timeout=30.0) as client:
        for endpoint in test_endpoints:
            url = f"{base_url}{endpoint}"
            try:
                print(f"Testing {url}...")
                response = await client.get(url, headers=headers)
                print(f"  Status: {response.status_code}")
                if response.status_code < 400:
                    print(f"  Response: {response.text[:200]}...")
                elif response.status_code == 401:
                    print(f"  Unauthorized (but endpoint exists)")
                elif response.status_code == 404:
                    print(f"  Not found")
                else:
                    print(f"  Other error: {response.status_code}")
            except Exception as e:
                print(f"  Error: {e}")
            print()

        # Also try to get API documentation
        try:
            print("Testing for API documentation...")
            docs_endpoints = ["/docs", "/v1/docs", "/swagger", "/openapi.json"]
            for doc_endpoint in docs_endpoints:
                url = f"{base_url}{doc_endpoint}"
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        print(f"Found docs at {url}")
                        print(f"Content type: {response.headers.get('content-type')}")
                except:
                    pass
        except Exception as e:
            print(f"Docs test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_tensorlake_endpoints())