#!/usr/bin/env python3
"""
Debug script to test the auth flow for agentic search.
This will help us understand what's happening in get_user_from_token_optimized.
"""

import asyncio
import httpx
from services.auth_utils import get_user_from_token_optimized
from memory.memory_graph import MemoryGraph
from models.memory_models import SearchRequest

async def test_auth_flow():
    """Test the auth flow directly to see what's happening"""
    
    print("🔍 Testing Auth Flow for Agentic Search")
    print("=" * 50)
    
    # Initialize MemoryGraph
    memory_graph = MemoryGraph()
    
    # Test parameters
    auth_header = "APIKey sk-proj-papr-test-key-mhnkVbAdgG-pohYfXWoOK"
    client_type = "papr_plugin"
    
    # Create search request with agentic graph enabled
    search_request = SearchRequest(
        query="Introducing Papr: Predictive Memory Layer that helps AI agents remember",
        rank_results=True,
        user_id="qQKa7NLSPm",
        enable_agentic_graph=True,
    )
    
    print(f"Auth Header: {auth_header}")
    print(f"Client Type: {client_type}")
    print(f"Search Request: {search_request.model_dump()}")
    print(f"Enable Agentic Graph: {search_request.enable_agentic_graph}")
    print("-" * 50)
    
    try:
        async with httpx.AsyncClient() as httpx_client:
            print("📡 Calling get_user_from_token_optimized...")
            
            auth_response = await get_user_from_token_optimized(
                auth_header=auth_header,
                client_type=client_type,
                memory_graph=memory_graph,
                search_request=search_request,
                httpx_client=httpx_client,
                include_schemas=False,
                url_enable_agentic_graph=None  # Test with None (should use JSON body value)
            )
            
            print("✅ Auth successful!")
            print(f"Developer ID: {auth_response.developer_id}")
            print(f"End User ID: {auth_response.end_user_id}")
            print(f"Workspace ID: {auth_response.workspace_id}")
            print(f"Is Qwen Route: {auth_response.is_qwen_route}")
            
            # Check if schema cache was created
            if hasattr(auth_response, 'schema_cache') and auth_response.schema_cache:
                schema_cache = auth_response.schema_cache
                print(f"✅ Schema Cache Created!")
                print(f"Schema Cache Keys: {list(schema_cache.keys())}")
                
                if 'patterns' in schema_cache:
                    patterns = schema_cache['patterns']
                    print(f"Patterns Count: {len(patterns)}")
                    if patterns:
                        print(f"First 3 Patterns: {patterns[:3]}")
                else:
                    print("❌ No 'patterns' key in schema cache")
                    
                if 'user_schemas' in schema_cache:
                    user_schemas = schema_cache['user_schemas']
                    print(f"User Schemas Count: {len(user_schemas)}")
                else:
                    print("ℹ️ No 'user_schemas' key in schema cache")
            else:
                print("❌ No Schema Cache Created")
                
    except Exception as e:
        print(f"❌ Auth failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Auth Debug Test")
    print("This will test the auth flow directly without needing a running server")
    print("=" * 60)
    
    asyncio.run(test_auth_flow())
    
    print("=" * 60)
    print("🏁 Test Complete!")
    print("\n📋 Check the console output above to see:")
    print("1. Whether auth succeeded")
    print("2. Whether schema cache was created")
    print("3. Whether patterns were found")
    print("4. Any error messages or debug logs")

