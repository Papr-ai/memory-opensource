#!/usr/bin/env python3
"""Simple test to check active patterns flow during search"""

import asyncio
import httpx

BASE_URL = "http://localhost:8000"
TEST_API_KEY = "f80c5a2940f21882420b41690522cb2c"

async def test_search_only():
    """Test search with agentic graph to see DEBUG logs"""
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": TEST_API_KEY
    }
    
    search_payload = {
        "query": "What security policies protect customer data?",
        "enable_agentic_graph": True,
        "rank_results": True,
        "external_user_id": "security_user_456"
    }
    
    print("\n" + "="*80)
    print("🔍 TESTING AGENTIC SEARCH WITH ACTIVE PATTERNS")
    print("="*80)
    print(f"Query: {search_payload['query']}")
    print(f"enable_agentic_graph: {search_payload['enable_agentic_graph']}")
    print(f"external_user_id: {search_payload['external_user_id']}")
    print("\nSending request to /v1/memory/search...")
    print("="*80 + "\n")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/memory/search",
            headers=headers,
            params={"max_memories": 20},
            json=search_payload
        )
        
        print(f"\n✅ Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {len(data.get('data', []))} results")
            print("\n" + "="*80)
            print("✅ SUCCESS! Now check the uvicorn server logs for:")
            print("="*80)
            print("  🔧 DEBUG AUTH: schema_cache_task...")
            print("  🔧 DEBUG: cached_schema type...")
            print("  🔧 Added X cached ActivePatterns... OR No cached_schema patterns...")
            print("="*80)
        else:
            print(f"❌ Search failed: {response.text[:500]}")

if __name__ == "__main__":
    asyncio.run(test_search_only())
