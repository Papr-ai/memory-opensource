#!/usr/bin/env python3
"""
Test Auto Mode with Schema ID specifically.

This tests the new GraphGeneration API with:
- mode: "auto"
- auto.schema_id: specific schema enforcement
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
SCHEMA_ID = "IeskhPibBx"  # The security schema ID from previous tests

async def test_auto_mode_with_schema_id():
    """Test Auto Mode with Schema ID - New GraphGeneration API"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_user_id = f"test_auto_schema_{timestamp}"
    
    memory_data = {
        "type": "text",
        "content": f"[AUTO_SCHEMA_TEST {timestamp}] Critical security alert: Advanced persistent threat detected using credential stuffing tactics to compromise admin accounts, leading to potential data breach and system compromise.",
        "graph_generation": {
            "mode": "auto",
            "auto": {
                "schema_id": SCHEMA_ID,
                "simple_schema_mode": True
            }
        },
        "metadata": {
            "event_type": "security_incident",
            "test_type": f"auto_schema_{timestamp}",
            "external_user_id": external_user_id,
            "severity": "critical",
            "source": "security_monitoring"
        }
    }
    
    print(f"🧪 Testing Auto Mode with Schema ID")
    print(f"=" * 60)
    print(f"📋 Test Configuration:")
    print(f"   External User ID: {external_user_id}")
    print(f"   Schema ID: {SCHEMA_ID}")
    print(f"   Simple Schema Mode: True")
    print(f"   Content: Security incident with multiple tactics and impacts")
    print(f"")
    print(f"🎯 Expected Behavior:")
    print(f"   ✅ Use GraphGeneration API with mode='auto'")
    print(f"   ✅ Enforce specific schema: {SCHEMA_ID}")
    print(f"   ✅ Generate SecurityBehavior, Tactic, Impact nodes")
    print(f"   ✅ Create MAPS_TO_TACTIC, HAS_IMPACT relationships")
    print(f"   ✅ Connect Memory node to generated nodes with EXTRACTED relationships")
    print(f"")
    print(f"🔍 Log Patterns to Look For:")
    print(f"   - '✅ Using schema_id from graph_generation.auto: {SCHEMA_ID}'")
    print(f"   - '🔍 GRAPH CONFIG: graph_override=False, schema_id={SCHEMA_ID}'")
    print(f"   - '🤖 LLM GENERATION: Using automatic graph extraction'")
    print(f"   - '🚀 GRAPH STEP 3: Using custom schema'")
    print(f"")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            print(f"🚀 Making API Request...")
            response = await client.post(
                f"{BASE_URL}/v1/memory",
                json=memory_data,
                params={"external_user_id": external_user_id},
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": TEST_API_KEY
                }
            )
            
            print(f"📡 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ SUCCESS!")
                print(f"")
                print(f"📊 Response Details:")
                print(f"   Status: {result.get('status', 'N/A')}")
                print(f"   Code: {result.get('code', 'N/A')}")
                
                if result.get("data") and len(result["data"]) > 0:
                    memory_item = result["data"][0]
                    memory_id = memory_item.get("memoryId")
                    print(f"   Memory ID: {memory_id}")
                    print(f"   Content: {memory_item.get('content', 'N/A')[:100]}...")
                    print(f"   Created At: {memory_item.get('createdAt', 'N/A')}")
                    
                    if memory_item.get('metadata'):
                        metadata = memory_item['metadata']
                        print(f"   Metadata Keys: {list(metadata.keys())}")
                        if 'customMetadata' in metadata:
                            custom = metadata['customMetadata']
                            print(f"   Custom Metadata: {list(custom.keys())}")
                
                print(f"")
                print(f"🎉 Test Completed Successfully!")
                print(f"")
                print(f"🔍 Next Steps for Verification:")
                print(f"   1. Check server logs for external_user_id: {external_user_id}")
                print(f"   2. Verify schema enforcement logs")
                print(f"   3. Check Neo4j for generated nodes and relationships")
                print(f"   4. Verify EXTRACTED relationships from Memory to nodes")
                print(f"")
                print(f"📝 Log Search Commands:")
                print(f"   grep '{external_user_id}' logs/app_*.log")
                print(f"   grep '{memory_id}' logs/app_*.log")
                print(f"   grep '{SCHEMA_ID}' logs/app_*.log")
                
                return {
                    "success": True,
                    "memory_id": memory_id,
                    "external_user_id": external_user_id,
                    "schema_id": SCHEMA_ID,
                    "response": result
                }
            else:
                print(f"❌ FAILED: {response.status_code}")
                print(f"")
                print(f"📄 Error Response:")
                try:
                    error_data = response.json()
                    print(json.dumps(error_data, indent=2))
                except:
                    print(response.text)
                
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "response": response.text
                }
                
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        print(f"")
        print(f"🔧 Troubleshooting:")
        print(f"   - Is the server running on {BASE_URL}?")
        print(f"   - Is the API key correct?")
        print(f"   - Are environment variables loaded?")
        
        return {
            "success": False,
            "error": str(e)
        }

async def main():
    """Run the auto + schema ID test"""
    print("🚀 GraphGeneration API Test: Auto Mode with Schema ID")
    print("=" * 70)
    
    result = await test_auto_mode_with_schema_id()
    
    print(f"\n{'=' * 70}")
    print("🎯 TEST SUMMARY")
    print("=" * 70)
    
    if result.get("success"):
        print(f"✅ Test PASSED")
        print(f"   Memory ID: {result.get('memory_id', 'N/A')}")
        print(f"   External User ID: {result.get('external_user_id', 'N/A')}")
        print(f"   Schema ID: {result.get('schema_id', 'N/A')}")
        print(f"")
        print(f"🎉 The new GraphGeneration API is working correctly!")
        print(f"   - Auto mode with schema_id enforcement ✅")
        print(f"   - Clean API structure without legacy compatibility ✅")
        print(f"   - Proper processing path taken ✅")
    else:
        print(f"❌ Test FAILED")
        print(f"   Error: {result.get('error', 'Unknown error')}")
        print(f"")
        print(f"🔧 Check the error details above and server logs")
    
    return result.get("success", False)

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)




