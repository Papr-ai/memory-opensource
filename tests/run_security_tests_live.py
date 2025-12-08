#!/usr/bin/env python3
"""
Run security schema tests against a running server at localhost:8000

This script runs only the security schema tests against a live server.
Make sure the server is running before executing this script:
  source .env && poetry run uvicorn main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import sys
import time
import httpx
from pathlib import Path

# Add the project root and tests directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from tests.test_security_schema_v1 import (
    get_security_schema_data,
    TEST_API_KEY,
)

BASE_URL = "http://localhost:8000"

# Test state to share data between tests
_test_state = {}


async def check_server():
    """Check if server is running"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BASE_URL}/health")
            return response.status_code == 200
    except Exception:
        return False


async def test_create_security_schema():
    """Test 1: Create custom security schema"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        schema_data = get_security_schema_data()
        _test_state["schema_data"] = schema_data

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": TEST_API_KEY
        }

        response = await client.post(f"{BASE_URL}/v1/schemas", headers=headers, json=schema_data)

        if response.status_code != 201:
            raise Exception(f"Schema creation failed: {response.text}")

        result = response.json()
        schema_id = result["data"]["id"]
        _test_state["schema_id"] = schema_id

        print(f"   ✅ Schema created: {schema_id}")
        return True


async def test_add_memory_with_schema_id():
    """Test 2: Add memory with schema_id"""
    if "schema_id" not in _test_state:
        raise Exception("Schema must be created first")

    async with httpx.AsyncClient(timeout=30.0) as client:
        schema_id = _test_state["schema_id"]

        memory_data = {
            "content": "Security incident detected: Unauthorized access attempt from IP 192.168.1.100 targeting admin panel. Multiple failed login attempts detected.",
            "type": "text",
            "metadata": {
                "schema_id": schema_id,
                "event_type": "security_incident"
            }
        }

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": TEST_API_KEY
        }

        response = await client.post(f"{BASE_URL}/v1/memory", headers=headers, json=memory_data)

        if response.status_code != 200:
            raise Exception(f"Memory creation failed: {response.text}")

        result = response.json()

        # Use the proper response structure: AddMemoryResponse.data[0].memoryId
        if not result.get("data") or len(result["data"]) == 0:
            raise Exception(f"Response missing data field or empty: {result}")

        memory_id = result["data"][0]["memoryId"]
        _test_state["memory_id"] = memory_id

        print(f"   ✅ Memory created: {memory_id}")
        return True


async def test_wait_for_processing():
    """Test 3: Wait for memory processing"""
    if "memory_id" not in _test_state:
        raise Exception("Memory must be created first")

    memory_id = _test_state["memory_id"]
    print(f"   ⏳ Waiting for memory {memory_id} to process...")

    async with httpx.AsyncClient(timeout=240.0) as client:
        headers = {"X-API-Key": TEST_API_KEY}

        for i in range(120):  # Wait up to 240 seconds
            await asyncio.sleep(2)

            response = await client.get(
                f"{BASE_URL}/v1/memory/{memory_id}",
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                processing_status = data.get("processing_status") or data.get("processingStatus")

                if processing_status == "completed":
                    print(f"   ✅ Memory processing completed")
                    return True
                elif processing_status == "failed":
                    raise Exception(f"Memory processing failed")

            if i % 5 == 0:
                print(f"   ⏳ Still waiting... ({i*2}s)")

        raise Exception("Timeout waiting for memory processing")


async def test_search_and_verify():
    """Test 4: Search and verify Neo4j nodes"""
    if "memory_id" not in _test_state:
        raise Exception("Memory must be created first")

    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {"X-API-Key": TEST_API_KEY}

        search_data = {
            "query": "security incident unauthorized access",
            "top_k": 5
        }

        response = await client.post(
            f"{BASE_URL}/v1/memory/search",
            headers=headers,
            json=search_data
        )

        if response.status_code != 200:
            raise Exception(f"Search failed: {response.text}")

        result = response.json()
        memories = result.get("memories", [])

        if not memories:
            raise Exception("No memories found in search results")

        print(f"   ✅ Found {len(memories)} memories")
        return True


async def test_agentic_search():
    """Test 5: Search with agentic graph"""
    if "schema_id" not in _test_state:
        raise Exception("Schema must be created first")

    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {"X-API-Key": TEST_API_KEY}

        search_data = {
            "query": "What security behaviors were detected?",
            "top_k": 5,
            "enable_agentic_graph": True
        }

        response = await client.post(
            f"{BASE_URL}/v1/memory/search",
            headers=headers,
            json=search_data
        )

        if response.status_code != 200:
            raise Exception(f"Agentic search failed: {response.text}")

        result = response.json()
        print(f"   ✅ Agentic search completed")
        return True


async def main():
    """Run all security tests"""
    print("🚀 Security Schema Tests - Live Server Mode")
    print("=" * 60)
    print(f"Testing against: {BASE_URL}")
    print("=" * 60)

    # Check if server is running
    print("\n🔍 Checking server status...")
    if not await check_server():
        print(f"❌ Server not responding at {BASE_URL}")
        print("\nPlease start the server first:")
        print("  cd /Users/amirkabbara/Documents/GitHub/memory")
        print("  source .env && poetry run uvicorn main:app --host 0.0.0.0 --port 8000")
        return False

    print(f"✅ Server is running\n")

    tests = [
        ("Create Security Schema", test_create_security_schema),
        ("Add Memory with schema_id", test_add_memory_with_schema_id),
        ("Wait for Memory Processing", test_wait_for_processing),
        ("Search and Verify Neo4j", test_search_and_verify),
        ("Agentic Graph Search", test_agentic_search),
    ]

    passed = 0
    failed = 0
    start_time = time.time()

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        try:
            await test_func()
            print(f"✅ Test passed: {test_name}")
            passed += 1
        except Exception as e:
            print(f"❌ Test failed: {test_name}")
            print(f"   Error: {str(e)}")
            failed += 1

    end_time = time.time()
    duration = end_time - start_time

    print(f"\n{'='*60}")
    print(f"Test Summary")
    print(f"{'='*60}")
    print(f"Total Tests: {passed + failed}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Duration: {duration:.2f}s")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        if success:
            print("\n✅ All security schema tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
