#!/usr/bin/env python3
"""
Quick Schema Test - Simple version for immediate testing

This script tests the schema functionality step by step with clear instructions.
"""

import asyncio
import httpx
import json
import os
from dotenv import find_dotenv, load_dotenv

# Load environment variables from .env file
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

# Configuration - Load from environment variables
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")  # Default to localhost:8000
API_KEY = os.getenv("TEST_X_USER_API_KEY")  # API key from environment
SESSION_TOKEN = os.getenv("TEST_SESSION_TOKEN")  # Session token from environment

async def test_schema_creation():
    """Test creating a schema"""
    print("🏗️  Testing Schema Creation")
    print("-" * 40)
    
    headers = {
        "Content-Type": "application/json",
        "X-Client-Type": "papr_plugin",
        "X-API-Key": API_KEY,
        "X-Session-Token": SESSION_TOKEN
    }
    
    # Simple e-commerce schema
    schema_data = {
        "name": "Simple E-commerce Schema",
        "description": "Basic schema for testing",
        "status": "active",
        "node_types": {
            "Product": {
                "name": "Product",
                "label": "Product",
                "properties": {
                    "name": {"type": "string", "required": True},
                    "price": {"type": "float", "required": True}
                },
                "required_properties": ["name", "price"]
            },
            "Customer": {
                "name": "Customer", 
                "label": "Customer",
                "properties": {
                    "name": {"type": "string", "required": True},
                    "email": {"type": "string", "required": True}
                },
                "required_properties": ["name", "email"]
            }
        },
        "relationship_types": {
            "PURCHASED": {
                "name": "PURCHASED",
                "label": "Purchased",
                "allowed_source_types": ["Customer"],
                "allowed_target_types": ["Product"]
            }
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            print(f"📤 Creating schema: {schema_data['name']}")
            
            response = await client.post(
                f"{BASE_URL}/v1/schemas",
                headers=headers,
                json=schema_data
            )
            
            print(f"📥 Response: {response.status_code}")
            
            if response.status_code == 201:
                result = response.json()
                if result.get("success"):
                    schema_id = result["data"]["id"]
                    print(f"✅ Schema created successfully!")
                    print(f"   ID: {schema_id}")
                    return schema_id
                else:
                    print(f"❌ Failed: {result.get('error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    return None

async def test_memory_addition():
    """Test adding a memory that should use the schema"""
    print("\n📝 Testing Memory Addition")
    print("-" * 40)
    
    headers = {
        "Content-Type": "application/json",
        "X-Client-Type": "papr_plugin", 
        "X-API-Key": API_KEY,
        "X-Session-Token": SESSION_TOKEN
    }
    
    # Content that should trigger e-commerce schema
    memory_data = {
        "content": "Customer John Smith purchased iPhone 15 for $999 from our store. His email is john@example.com.",
        "type": "text"
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            print(f"📤 Adding memory with e-commerce content")
            print(f"   Content: {memory_data['content']}")
            
            response = await client.post(
                f"{BASE_URL}/v1/memories",
                headers=headers,
                json=memory_data
            )
            
            print(f"📥 Response: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    memory_id = result["data"][0]["id"]
                    print(f"✅ Memory added successfully!")
                    print(f"   ID: {memory_id}")
                    print(f"   🧠 GPT-5-mini should have selected E-commerce Schema")
                    return memory_id
                else:
                    print(f"❌ Failed: {result.get('error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    return None

async def test_memory_search():
    """Test searching memories with schema"""
    print("\n🔍 Testing Memory Search")
    print("-" * 40)
    
    headers = {
        "Content-Type": "application/json",
        "X-Client-Type": "papr_plugin",
        "X-API-Key": API_KEY,
        "X-Session-Token": SESSION_TOKEN
    }
    
    # Search query that should use e-commerce schema
    search_data = {
        "query": "find customers who purchased iPhone products",
        "max_memories": 5
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            print(f"📤 Searching with e-commerce query")
            print(f"   Query: {search_data['query']}")
            
            response = await client.post(
                f"{BASE_URL}/v1/memories/search",
                headers=headers,
                json=search_data
            )
            
            print(f"📥 Response: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    memories = result.get("data", {}).get("memories", [])
                    nodes = result.get("data", {}).get("nodes", [])
                    
                    print(f"✅ Search successful!")
                    print(f"   Found {len(memories)} memories")
                    print(f"   Found {len(nodes)} graph nodes")
                    print(f"   🧠 GPT-5-mini should have used E-commerce Schema")
                    
                    if memories:
                        print(f"   First result: {memories[0].get('content', '')[:60]}...")
                    
                    return True
                else:
                    print(f"❌ Failed: {result.get('error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    return False

async def main():
    """Run the quick test"""
    print("🚀 Quick Schema Test")
    print("=" * 50)
    
    # Check configuration
    if not API_KEY:
        print("❌ TEST_X_USER_API_KEY not set in environment variables")
        print("   Please set it in your .env file or export it")
        return
    
    if not SESSION_TOKEN:
        print("❌ TEST_SESSION_TOKEN not set in environment variables")
        print("   Please set it in your .env file or export it")
        return
    
    print(f"🔧 Testing against: {BASE_URL}")
    print(f"🔑 Using API Key: {API_KEY[:10]}..." if API_KEY else "🔑 Using API Key: (not set)")
    
    # Run tests
    print("\n" + "="*50)
    schema_id = await test_schema_creation()
    
    if schema_id:
        print("\n⏳ Waiting 2 seconds for schema processing...")
        await asyncio.sleep(2)
        
        memory_id = await test_memory_addition()
        
        if memory_id:
            print("\n⏳ Waiting 3 seconds for memory processing...")
            await asyncio.sleep(3)
            
            search_success = await test_memory_search()
            
            # Summary
            print("\n" + "="*50)
            print("📊 TEST SUMMARY")
            print("="*50)
            print(f"Schema Creation: {'✅ PASS' if schema_id else '❌ FAIL'}")
            print(f"Memory Addition: {'✅ PASS' if memory_id else '❌ FAIL'}")
            print(f"Memory Search:   {'✅ PASS' if search_success else '❌ FAIL'}")
            
            if schema_id and memory_id and search_success:
                print("\n🎉 ALL TESTS PASSED!")
                print("✅ Your schema system is working correctly")
            else:
                print("\n⚠️  Some tests failed - check the logs above")
        else:
            print("\n❌ Memory addition failed - skipping search test")
    else:
        print("\n❌ Schema creation failed - cannot continue")

if __name__ == "__main__":
    print("📋 Instructions:")
    print("1. Make sure your API server is running")
    print("2. Set TEST_X_USER_API_KEY and TEST_SESSION_TOKEN in your .env file")
    print("3. Optionally set BASE_URL in .env (defaults to http://localhost:8000)")
    print("4. Run: python quick_schema_test.py")
    print()
    
    asyncio.run(main())







