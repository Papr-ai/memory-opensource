#!/usr/bin/env python3
"""
Test script to verify the new GraphGeneration API works correctly.

This tests the new nested structure:
- graph_generation.mode: "auto" | "manual"
- graph_generation.auto: { schema_id, simple_schema_mode, property_overrides }
- graph_generation.manual: { nodes, relationships }
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

async def test_auto_mode_default():
    """Test 1: Auto mode with default settings (pure AI)"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_user_id = f"test_auto_default_{timestamp}"
    
    memory_data = {
        "type": "text",
        "content": f"[AUTO_DEFAULT_TEST {timestamp}] John completed the quarterly report for Project Alpha.",
        "metadata": {
            "event_type": "task_completion",
            "test_type": f"auto_default_{timestamp}",
            "external_user_id": external_user_id
        }
        # No graph_generation field = defaults to auto mode with AI selection
    }
    
    print(f"🧪 Test 1: Auto Mode Default (Pure AI)")
    print(f"   External User ID: {external_user_id}")
    print(f"   Expected: AI selects schema and generates all properties")
    
    return await make_memory_request(memory_data, external_user_id, "auto_default")

async def test_auto_mode_with_schema():
    """Test 2: Auto mode with specific schema"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_user_id = f"test_auto_schema_{timestamp}"
    
    memory_data = {
        "type": "text",
        "content": f"[AUTO_SCHEMA_TEST {timestamp}] Security alert: SQL injection detected on /api/users endpoint.",
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
            "external_user_id": external_user_id
        }
    }
    
    print(f"🧪 Test 2: Auto Mode with Schema")
    print(f"   External User ID: {external_user_id}")
    print(f"   Schema ID: {SCHEMA_ID}")
    print(f"   Simple Schema Mode: True")
    print(f"   Expected: AI uses specified schema in simple mode")
    
    return await make_memory_request(memory_data, external_user_id, "auto_schema")

async def test_auto_mode_with_property_overrides():
    """Test 3: Auto mode with property overrides"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_user_id = f"test_auto_props_{timestamp}"
    
    memory_data = {
        "type": "text",
        "content": f"[AUTO_PROPS_TEST {timestamp}] John completed the quarterly report for Project Alpha.",
        "graph_generation": {
            "mode": "auto",
            "auto": {
                "property_overrides": {
                    "Person": {"id": f"person_john_{timestamp}"},
                    "Project": {"id": f"project_alpha_{timestamp}"}
                }
            }
        },
        "metadata": {
            "event_type": "task_completion",
            "test_type": f"auto_props_{timestamp}",
            "external_user_id": external_user_id
        }
    }
    
    print(f"🧪 Test 3: Auto Mode with Property Overrides")
    print(f"   External User ID: {external_user_id}")
    print(f"   Property Overrides: Person.id, Project.id")
    print(f"   Expected: AI generates graph but uses specified property values")
    
    return await make_memory_request(memory_data, external_user_id, "auto_props")

async def test_manual_mode():
    """Test 4: Manual mode with complete graph specification"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_user_id = f"test_manual_{timestamp}"
    
    memory_data = {
        "type": "text",
        "content": f"[MANUAL_TEST {timestamp}] Reference data for structured import.",
        "graph_generation": {
            "mode": "manual",
            "manual": {
                "nodes": [
                    {
                        "id": f"person_{timestamp}",
                        "label": "Person",
                        "properties": {
                            "name": "John Doe",
                            "role": "Manager"
                        }
                    },
                    {
                        "id": f"task_{timestamp}",
                        "label": "Task", 
                        "properties": {
                            "title": "Quarterly Report",
                            "status": "completed"
                        }
                    }
                ],
                "relationships": [
                    {
                        "source_node_id": f"person_{timestamp}",
                        "target_node_id": f"task_{timestamp}",
                        "relationship_type": "COMPLETED"
                    }
                ]
            }
        },
        "metadata": {
            "event_type": "data_import",
            "test_type": f"manual_{timestamp}",
            "external_user_id": external_user_id
        }
    }
    
    print(f"🧪 Test 4: Manual Mode")
    print(f"   External User ID: {external_user_id}")
    print(f"   Nodes: 2 (Person, Task)")
    print(f"   Relationships: 1 (COMPLETED)")
    print(f"   Expected: Exact graph structure created, no AI processing")
    
    return await make_memory_request(memory_data, external_user_id, "manual")

async def make_memory_request(memory_data, external_user_id, test_type):
    """Helper function to make memory request and handle response"""
    try:
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
                print(f"   ✅ SUCCESS!")
                if result.get("data") and len(result["data"]) > 0:
                    memory_id = result["data"][0].get("memoryId")
                    print(f"   Memory ID: {memory_id}")
                    return {"success": True, "memory_id": memory_id, "test_type": test_type}
                return {"success": True, "test_type": test_type}
            else:
                print(f"   ❌ FAILED: {response.status_code}")
                print(f"   Response: {response.text}")
                return {"success": False, "error": response.text, "test_type": test_type}
                
    except Exception as e:
        print(f"   ❌ EXCEPTION: {str(e)}")
        return {"success": False, "error": str(e), "test_type": test_type}


async def main():
    """Run all GraphGeneration API tests"""
    print("🚀 GraphGeneration API Tests")
    print("=" * 60)
    print("Testing the new graph_generation.mode structure:")
    print("- Auto mode: AI-powered with optional guidance")
    print("- Manual mode: Complete developer control")
    print("=" * 60)
    
    tests = [
        ("Auto Mode Default (Pure AI)", test_auto_mode_default),
        ("Auto Mode with Schema", test_auto_mode_with_schema),
        ("Auto Mode with Property Overrides", test_auto_mode_with_property_overrides),
        ("Manual Mode", test_manual_mode),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-' * 60}")
        try:
            result = await test_func()
            results.append(result)
        except Exception as e:
            print(f"   ❌ TEST FAILED: {str(e)}")
            results.append({"success": False, "error": str(e), "test_type": test_name})
    
    # Summary
    print(f"\n{'=' * 60}")
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    
    if failed:
        print(f"\n❌ Failed Tests:")
        for result in failed:
            print(f"   - {result.get('test_type', 'Unknown')}: {result.get('error', 'Unknown error')}")
    
    if successful:
        print(f"\n✅ Successful Tests:")
        for result in successful:
            memory_id = result.get('memory_id', 'N/A')
            print(f"   - {result.get('test_type', 'Unknown')}: {memory_id}")
    
    print(f"\n📋 Check logs for detailed processing information")
    print(f"🔍 Search logs for memory IDs to verify graph generation")
    
    return len(failed) == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
