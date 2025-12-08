#!/usr/bin/env python3
"""
Test if relationships are now being generated with our fix
"""

import asyncio
import httpx
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"
TEST_API_KEY = "f80c5a2940f21882420b41690522cb2c"
SCHEMA_ID = "IeskhPibBx"

# Use a unique timestamp to track this specific test
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXTERNAL_USER_ID = f"test_relationships_{TIMESTAMP}"

async def create_memory_with_relationships():
    """Create a memory and check if relationships are generated"""
    
    memory_data = {
        "type": "text",
        "content": f"[TEST {TIMESTAMP}] Critical security alert: Advanced persistent threat detected. Attackers used credential stuffing to compromise admin accounts, then deployed ransomware affecting data integrity across multiple systems.",
        "schema_id": SCHEMA_ID,
        "metadata": {
            "event_type": "security_incident",
            "test_type": f"relationship_test_{TIMESTAMP}",
            "external_user_id": EXTERNAL_USER_ID
        }
    }
    
    print(f"🚀 Testing Relationship Generation Fix")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"External User ID: {EXTERNAL_USER_ID}")
    print(f"Schema ID: {SCHEMA_ID}")
    print("="*60)
    
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
            if result.get("data") and len(result["data"]) > 0:
                memory_id = result["data"][0].get("memoryId")
                print(f"Memory ID: {memory_id}")
                print(f"Now check logs for: {memory_id}")
                print(f"Look for: SecurityBehavior, Tactic, Impact nodes")
                print(f"Look for: MAPS_TO_TACTIC, HAS_IMPACT relationships")
                return memory_id
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None

async def main():
    start_time = time.time()
    memory_id = await create_memory_with_relationships()
    end_time = time.time()
    
    print(f"\n⏱️  Total time: {end_time - start_time:.2f}s")
    if memory_id:
        print(f"\n📝 To check results, search logs for:")
        print(f"   Memory ID: {memory_id}")
        print(f"   Test ID: {TIMESTAMP}")
        print(f"   External User: {EXTERNAL_USER_ID}")

if __name__ == "__main__":
    asyncio.run(main())




