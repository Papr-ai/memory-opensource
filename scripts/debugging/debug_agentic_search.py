#!/usr/bin/env python3
"""
Debug agentic search to see why 0 nodes are returned
"""
import asyncio
import httpx
import json
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

async def debug_agentic_search():
    """Debug agentic search with detailed logging"""
    
    BASE_URL = "http://localhost:8000"
    
    # Get API key and session token from environment variables
    api_key = os.getenv("TEST_X_USER_API_KEY")
    session_token = os.getenv("TEST_SESSION_TOKEN")
    
    if not api_key:
        raise ValueError("TEST_X_USER_API_KEY environment variable is required")
    if not session_token:
        raise ValueError("TEST_SESSION_TOKEN environment variable is required")
    
    HEADERS = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
        "X-Session-Token": session_token
    }
    
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        print("🔍 Testing agentic search with debug info...")
        
        search_request = {
            "query": "Python FastAPI functions created by Jennifer Park",
            # Don't hardcode user_id - let the system resolve it from session token like memory addition does
            "enable_agentic_graph": True,
            "rank_results": True
            # Temporarily remove external_user_id to test dynamic schema improvements
        }
        
        print(f"🔍 Search Query: {search_request['query']}")
        print(f"🔍 Agentic graph enabled: {search_request['enable_agentic_graph']}")
        print(f"🔍 JSON Body: {json.dumps(search_request, indent=2)}")
        
        response = await client.post(
            "/v1/memory/search",
            params={
                "max_memories": 20,
                "max_nodes": 15,
                "enable_agentic_graph": True  # Also in URL params
            },
            headers=HEADERS,
            json=search_request
        )
        
        print(f"\n📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Check memories
            memories = result.get("data", {}).get("memories", [])
            print(f"📋 Memories found: {len(memories)}")
            
            # Check graph nodes (this is the key test)
            nodes = result.get("data", {}).get("nodes", [])
            print(f"🏗️ Graph nodes found: {len(nodes)}")
            
            # Check debug info if available
            debug_info = result.get("data", {}).get("debug_agentic_params")
            if debug_info:
                print(f"🔍 Debug info: {json.dumps(debug_info, indent=2)}")
            
            # Check for any agentic-related fields
            data_keys = list(result.get("data", {}).keys())
            print(f"🔍 Response data keys: {data_keys}")
            
            if len(nodes) == 0:
                print("❌ ISSUE: Agentic search returned 0 nodes")
                print("This suggests either:")
                print("  1. skip_neo=True (agentic search disabled)")
                print("  2. Neo4j query returned no results")
                print("  3. Cache lookup failed")
                print("  4. Pattern selection failed")
                
                # Print full response for debugging
                print(f"\n🔍 Full response structure:")
                print(json.dumps(result, indent=2)[:1000] + "..." if len(json.dumps(result)) > 1000 else json.dumps(result, indent=2))
            else:
                print("✅ SUCCESS: Agentic search is working!")
                for i, node in enumerate(nodes[:3]):
                    label = node.get('label', 'Unknown')
                    props = node.get('properties', {})
                    name = props.get('name', props.get('title', 'No name'))
                    print(f"  {i+1}. {label}: {name}")
        else:
            print(f"❌ Search failed with status {response.status_code}")
            print(f"Response: {response.text}")

if __name__ == "__main__":
    asyncio.run(debug_agentic_search())
