#!/usr/bin/env python3
"""
Test script to verify the Memory node fixes:
1. Memory nodes are not passed to LLM for relationship generation
2. Automatic CONTAINS relationships are created from Memory to generated nodes
3. Schema usage tracking works
"""

import httpx
import asyncio
import os
import json
import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

BASE_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8000")
TEST_API_KEY = os.getenv("PAPR_API_KEY", "f80c5a2940f21882420b41690522cb2c")
SCHEMA_ID = "IeskhPibBx"  # The security schema ID

async def test_memory_fixes() -> str:
    """Test the memory node fixes with a unique test case"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_user_id = f"test_memory_fixes_{timestamp}"
    test_content = f"[MEMORY_FIX_TEST {timestamp}] Security breach detected: Phishing attack compromised user credentials, leading to unauthorized data access and potential compliance violations."

    memory_data = {
        "type": "text",
        "content": test_content,
        "schema_id": SCHEMA_ID,
        "metadata": {
            "event_type": "security_incident",
            "test_type": f"memory_fixes_test_{timestamp}",
            "external_user_id": external_user_id
        }
    }

    print(f"🧪 Testing Memory Node Fixes")
    print(f"   Content: {memory_data['content'][:80]}...")
    print(f"   Schema ID: {SCHEMA_ID}")
    print(f"   External User ID: {external_user_id}")
    print(f"   Expected fixes:")
    print(f"     ✅ Memory nodes NOT passed to LLM for relationships")
    print(f"     ✅ Automatic CONTAINS relationships created")
    print(f"     ✅ Schema usage tracking increments usage_count")

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/memory",
            json=memory_data,
            params={"external_user_id": external_user_id},
            headers={
                "Content-Type": "application/json",
                "X-API-Key": TEST_API_KEY
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Memory created successfully!")
            print(f"Response: {result}")
            if result.get("data") and len(result["data"]) > 0:
                memory_id = result["data"][0].get("memoryId")
                print(f"Memory ID: {memory_id}")
                return memory_id
            return "success"
        else:
            print(f"\n❌ Failed to create memory: {response.status_code}")
            print(f"Response: {response.text}")
            return None

async def main():
    memory_id = await test_memory_fixes()
    if memory_id:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n🎉 SUCCESS!")
        print(f"   Memory ID: {memory_id}")
        print(f"   Schema ID: {SCHEMA_ID}")
        print(f"   Test completed at: {timestamp}")
        print(f"\n📋 Check logs for:")
        print(f"   🔍 'Generated nodes for LLM relationship generation' (should NOT include Memory)")
        print(f"   🔍 'Memory node (for context only)' (should be separate)")
        print(f"   🔍 'AUTO-CONNECT: Creating automatic relationships' (should create CONTAINS)")
        print(f"   🔍 'Schema usage updated' (should increment usage_count)")
        print(f"\n📊 Check Parse dashboard for schema usage_count increment")
    else:
        print(f"\n❌ FAILED to create memory with fixes.")

if __name__ == "__main__":
    asyncio.run(main())




