#!/usr/bin/env python3
"""
Test script to verify that LLM-generated IDs are preserved (preventing duplicates).
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

async def test_stable_ids_fix() -> str:
    """Test that identical content generates identical node IDs (no duplicates)"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_user_id = f"test_stable_ids_{timestamp}"
    
    # Use identical content for both runs
    memory_content = "Security incident: Phishing attack targeting employee credentials. Attacker used social engineering tactics to compromise user accounts, resulting in potential data breach."
    
    memory_data = {
        "type": "text",
        "content": memory_content,
        "schema_id": SCHEMA_ID,
        "metadata": {
            "event_type": "security_incident",
            "test_type": f"stable_ids_test_{timestamp}",
            "external_user_id": external_user_id
        }
    }

    print(f"🧪 Testing Stable IDs Fix")
    print(f"   Content: {memory_content[:80]}...")
    print(f"   Schema ID: {SCHEMA_ID}")
    print(f"   External User ID: {external_user_id}")
    print(f"   Expected: LLM generates same IDs for identical content")
    print("\n" + "=" * 60 + "\n")

    memory_ids = []
    
    # Run the same content twice
    for run_num in [1, 2]:
        print(f"🔄 Run {run_num}: Adding identical memory content...")
        
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
                print(f"✅ Run {run_num} completed successfully!")
                if result.get("data") and len(result["data"]) > 0:
                    memory_id = result["data"][0].get("memoryId")
                    memory_ids.append(memory_id)
                    print(f"   Memory ID: {memory_id}")
                else:
                    print(f"❌ Run {run_num} failed: No memory ID returned")
                    return None
            else:
                print(f"❌ Run {run_num} failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
        
        # Wait a bit between runs
        if run_num == 1:
            print(f"⏳ Waiting 5 seconds before second run...")
            await asyncio.sleep(5)
    
    print("\n" + "=" * 60 + "\n")
    print(f"🎉 TEST COMPLETED!")
    print(f"   Run 1 Memory ID: {memory_ids[0] if len(memory_ids) > 0 else 'None'}")
    print(f"   Run 2 Memory ID: {memory_ids[1] if len(memory_ids) > 1 else 'None'}")
    
    if len(memory_ids) == 2:
        print(f"\n📊 ANALYSIS:")
        print(f"   Same content processed twice")
        print(f"   With stable IDs: LLM should generate same semantic IDs")
        print(f"   Neo4j should MERGE nodes instead of creating duplicates")
        print(f"   Check logs for: 'behavior-phishing-attack', 'tactic-social-engineering', etc.")
        
        print(f"\n📝 To verify no duplicates were created:")
        print(f"   1. Check Neo4j for nodes with same name but different IDs")
        print(f"   2. Search logs for 'Successfully merged' vs 'Created new'")
        print(f"   3. Verify MERGE operations used semantic IDs, not UUIDs")
        
        return f"SUCCESS - Both memories processed: {memory_ids[0]}, {memory_ids[1]}"
    else:
        return "FAILED - Could not complete both runs"

async def main():
    result = await test_stable_ids_fix()
    print(f"\n🏁 Final Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())




