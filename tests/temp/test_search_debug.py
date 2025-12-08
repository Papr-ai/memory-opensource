#!/usr/bin/env python3
"""
Debug script to test search with agentic graph enabled.
This will help us understand what's happening in the auth flow.
"""

import asyncio
import httpx
import json
import os
from dotenv import load_dotenv

# Load environment variables
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
    if use_dotenv:
        load_dotenv()

async def test_search_with_agentic_graph():
    """Test search with agentic graph enabled to debug auth flow"""
    
    # Test configuration
    base_url = "http://localhost:8000"
    api_key = "sk-proj-papr-test-key-mhnkVbAdgG-pohYfXWoOK"  # Test API key from the test
    target_user_id = "qQKa7NLSPm"  # Test user ID
    
    headers = {
        'Content-Type': 'application/json',
        'X-Client-Type': 'papr_plugin',
        'X-API-Key': api_key,
        'Accept-Encoding': 'gzip'
    }
    
    # Search request with agentic graph enabled
    search_request = {
        "query": "Introducing Papr: Predictive Memory Layer that helps AI agents remember",
        "rank_results": True,
        "user_id": target_user_id,
        "enable_agentic_graph": True,
    }
    
    print("🔍 Testing Search with Agentic Graph Enabled")
    print(f"Base URL: {base_url}")
    print(f"API Key: {api_key}")
    print(f"User ID: {target_user_id}")
    print(f"Search Request: {json.dumps(search_request, indent=2)}")
    print("-" * 50)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # Test 1: Search with agentic graph in JSON body
            print("📡 Test 1: Agentic graph in JSON body")
            response = await client.post(
                f"{base_url}/v1/memory/search?max_memories=20&max_nodes=10",
                json=search_request,
                headers=headers
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success!")
                print(f"Memory Count: {len(result.get('data', {}).get('memories', []))}")
                print(f"Node Count: {len(result.get('data', {}).get('nodes', []))}")
                print(f"Search ID: {result.get('search_id', 'N/A')}")
            else:
                print(f"❌ Failed!")
                try:
                    error_data = response.json()
                    print(f"Error: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"Raw Response: {response.text}")
            
            print("-" * 50)
            
            # Test 2: Search with agentic graph in URL parameter
            print("📡 Test 2: Agentic graph in URL parameter")
            search_request_with_agentic = {
                "query": "Introducing Papr: Predictive Memory Layer that helps AI agents remember",
                "rank_results": True,
                "user_id": target_user_id,
                "enable_agentic_graph": True  # Now in JSON body only
            }
            
            response2 = await client.post(
                f"{base_url}/v1/memory/search?max_memories=20&max_nodes=10",
                json=search_request_with_agentic,
                headers=headers
            )
            
            print(f"Status Code: {response2.status_code}")
            
            if response2.status_code == 200:
                result2 = response2.json()
                print(f"✅ Success!")
                print(f"Memory Count: {len(result2.get('data', {}).get('memories', []))}")
                print(f"Node Count: {len(result2.get('data', {}).get('nodes', []))}")
                print(f"Search ID: {result2.get('search_id', 'N/A')}")
            else:
                print(f"❌ Failed!")
                try:
                    error_data2 = response2.json()
                    print(f"Error: {json.dumps(error_data2, indent=2)}")
                except:
                    print(f"Raw Response: {response2.text}")
            
            print("-" * 50)
            
            # Test 3: Search without agentic graph (control test)
            print("📡 Test 3: Search without agentic graph (control)")
            search_request_no_agentic = {
                "query": "Introducing Papr: Predictive Memory Layer that helps AI agents remember",
                "rank_results": True,
                "user_id": target_user_id,
                "enable_agentic_graph": False,
            }
            
            response3 = await client.post(
                f"{base_url}/v1/memory/search?max_memories=20&max_nodes=10",
                json=search_request_no_agentic,
                headers=headers
            )
            
            print(f"Status Code: {response3.status_code}")
            
            if response3.status_code == 200:
                result3 = response3.json()
                print(f"✅ Success!")
                print(f"Memory Count: {len(result3.get('data', {}).get('memories', []))}")
                print(f"Node Count: {len(result3.get('data', {}).get('nodes', []))}")
                print(f"Search ID: {result3.get('search_id', 'N/A')}")
            else:
                print(f"❌ Failed!")
                try:
                    error_data3 = response3.json()
                    print(f"Error: {json.dumps(error_data3, indent=2)}")
                except:
                    print(f"Raw Response: {response3.text}")
                    
        except Exception as e:
            print(f"❌ Exception occurred: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Search Debug Test")
    print("Make sure the server is running on http://localhost:8000")
    print("=" * 60)
    
    asyncio.run(test_search_with_agentic_graph())
    
    print("=" * 60)
    print("🏁 Test Complete!")
    print("\n📋 Next Steps:")
    print("1. Check the logs at /Users/amirkabbara/Documents/GitHub/memory/logs/app_2025-11-03.log")
    print("2. Look for '🔍 AUTH DEBUG' logs to see auth conditions")
    print("3. Look for '🚀 ENHANCED AGENTIC CACHE' logs to see if schema cache is created")
    print("4. Look for '🔍 CYPHER GENERATION DEBUG' logs to see if patterns are passed to LLM")

