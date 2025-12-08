#!/usr/bin/env python3
"""
Simple test: Add only a schema_id memory
"""

import asyncio
import httpx
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_API_KEY = "f80c5a2940f21882420b41690522cb2c"  # Use the working API key

# Use existing schema from successful test
SCHEMA_ID = "IeskhPibBx"  # From successful test
EXTERNAL_USER_ID = "security_test_user_004"  # New user to avoid conflicts


def get_memory_data() -> dict:
    """Get the memory data for schema_id approach"""
    return {
        "type": "text",
        "content": "Security incident detected: SQL injection attempt targeting /api/users endpoint from IP 192.168.1.100. This is a credential access tactic with high severity impact on data confidentiality.",
        "schema_id": SCHEMA_ID,
        "metadata": {
            "event_type": "security_incident",
            "test_type": "schema_id_approach_only",
            "external_user_id": EXTERNAL_USER_ID
        }
    }


async def add_memory_with_schema_id() -> str:
    """Add memory using schema_id approach"""
    print("📝 Adding memory with schema_id...")
    
    memory_data = get_memory_data()
    
    print(f"   Content: {memory_data['content'][:80]}...")
    print(f"   Schema ID: {memory_data['schema_id']}")
    print(f"   External User ID: {EXTERNAL_USER_ID}")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/memory",
            json=memory_data,
            params={"external_user_id": EXTERNAL_USER_ID},
            headers={
                "Content-Type": "application/json",
                "X-API-Key": TEST_API_KEY
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Memory created successfully!")
            print(f"Response: {result}")
            # Extract memory ID from the response data array
            if result.get("data") and len(result["data"]) > 0:
                memory_id = result["data"][0].get("memoryId")
                print(f"Memory ID: {memory_id}")
                return memory_id
            return "success"
        else:
            print(f"❌ Failed to create memory: {response.status_code}")
            print(f"Response: {response.text}")
            return None


async def main():
    """Run schema_id memory addition test"""
    print("🚀 Schema ID Memory Addition Test")
    print("=" * 60)
    print(f"Testing against: {BASE_URL}")
    print(f"Schema ID: {SCHEMA_ID}")
    print(f"External User ID: {EXTERNAL_USER_ID}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Add memory with schema_id
        memory_id = await add_memory_with_schema_id()
        
        if memory_id:
            print(f"\n✅ SUCCESS!")
            print(f"   Memory ID: {memory_id}")
            print(f"   Schema ID: {SCHEMA_ID}")
            print(f"   External User ID: {EXTERNAL_USER_ID}")
        else:
            print(f"\n❌ FAILED to create memory")
            return
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    end_time = time.time()
    print(f"\n⏱️  Total time: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
