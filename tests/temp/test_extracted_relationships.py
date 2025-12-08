#!/usr/bin/env python3
"""
Test script to verify EXTRACTED relationships with metadata are working correctly.
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

async def test_extracted_relationships() -> str:
    """Test the new EXTRACTED relationships with metadata"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_user_id = f"test_extracted_rel_{timestamp}"
    memory_content = f"[EXTRACTED_TEST {timestamp}] Critical security incident: SQL injection attack detected on production database. Attacker used credential stuffing to gain unauthorized access, compromising sensitive customer data and triggering compliance violations."

    memory_data = {
        "type": "text",
        "content": memory_content,
        "schema_id": SCHEMA_ID,
        "metadata": {
            "event_type": "security_incident",
            "test_type": f"extracted_relationships_test_{timestamp}",
            "external_user_id": external_user_id
        }
    }

    print(f"🧪 Testing EXTRACTED Relationships with Metadata")
    print(f"   Content: {memory_content[:80]}...")
    print(f"   Schema ID: {SCHEMA_ID}")
    print(f"   External User ID: {external_user_id}")
    print(f"   Expected changes:")
    print(f"     ✅ Relationships changed from CONTAINS to EXTRACTED")
    print(f"     ✅ Metadata added: extraction_method, extracted_at, schema_id")
    print(f"     ✅ Workspace and user isolation maintained")
    print("\n" + "=" * 60 + "\n")

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
            print(f"✅ Memory created successfully!")
            print(f"Response: {json.dumps(result, indent=2)}")
            if result.get("data") and len(result["data"]) > 0:
                memory_id = result["data"][0].get("memoryId")
                print(f"Memory ID: {memory_id}")
                print("\n" + "=" * 60 + "\n")
                print(f"🎉 SUCCESS!")
                print(f"   Memory ID: {memory_id}")
                print(f"   Schema ID: {SCHEMA_ID}")
                print(f"   Test completed at: {timestamp}")
                print("\n" + "=" * 60 + "\n")
                print(f"📋 Check logs for:")
                print(f"   🔍 'Creating automatic EXTRACTED relationships' (should show EXTRACTED not CONTAINS)")
                print(f"   🔍 'Created relationship EXTRACTED from {memory_id}' (should show EXTRACTED relationships)")
                print(f"   🔍 Relationship metadata: extraction_method, extracted_at, schema_id")
                print("\n📊 Neo4j Query to verify relationships:")
                print(f"   MATCH (m:Memory {{id: '{memory_id}'}})-[r:EXTRACTED]->(n)")
                print(f"   RETURN m, r, n")
                print(f"   // Should show EXTRACTED relationships with metadata properties")
                return memory_id
            return "success"
        else:
            print(f"❌ Failed to create memory: {response.status_code}")
            print(f"Response: {response.text}")
            return None

async def main():
    await test_extracted_relationships()

if __name__ == "__main__":
    asyncio.run(main())




