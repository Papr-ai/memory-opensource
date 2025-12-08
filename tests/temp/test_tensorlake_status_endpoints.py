#!/usr/bin/env python3
"""
Test TensorLake API status endpoints to find the correct one
"""

import asyncio
import httpx
import os
from dotenv import load_dotenv

async def test_status_endpoints():
    """Test various TensorLake status endpoints"""

    use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
    if use_dotenv:
        load_dotenv()

    api_key = os.getenv("TENSORLAKE_API_KEY")
    base_url = os.getenv("TENSORLAKE_BASE_URL", "https://api.tensorlake.ai")
    parse_id = "parse_RqW7rCqzpt6DmtWCMkTkf"  # From the previous test

    print(f"Testing TensorLake status endpoints...")
    print(f"Base URL: {base_url}")
    print(f"Parse ID: {parse_id}")

    headers = {"Authorization": f"Bearer {api_key}"}

    # Test various status endpoints
    test_endpoints = [
        f"/documents/v2/read/{parse_id}",
        f"/documents/v2/parse/{parse_id}",
        f"/documents/v2/status/{parse_id}",
        f"/documents/v2/jobs/{parse_id}",
        f"/documents/v2/result/{parse_id}",
        f"/documents/v1/parse/{parse_id}",
        f"/documents/v1/status/{parse_id}",
        f"/documents/v1/jobs/{parse_id}",
        f"/v2/read/{parse_id}",
        f"/v2/parse/{parse_id}",
        f"/v2/status/{parse_id}",
        f"/v2/jobs/{parse_id}",
        f"/parse/{parse_id}",
        f"/status/{parse_id}",
        f"/jobs/{parse_id}",
        f"/result/{parse_id}"
    ]

    async with httpx.AsyncClient(timeout=10.0) as client:
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
                    print(f"  Response: {response.text[:200]}...")
            except Exception as e:
                print(f"  Error: {e}")
            print()

if __name__ == "__main__":
    asyncio.run(test_status_endpoints())