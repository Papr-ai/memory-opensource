#!/usr/bin/env python3
"""
MongoDB Health Check Script

This script tests if MongoDB connection warmup is working properly by measuring
the performance of the first API key lookup after startup.

Usage:
    python scripts/mongodb_health_check.py
"""

import sys
import os
import time
import asyncio
import aiohttp
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from memory.memory_graph import MemoryGraph


async def test_cold_start_performance():
    """Test MongoDB performance from cold start"""
    print("🧪 Testing MongoDB Cold Start Performance")
    print("=" * 50)
    
    try:
        # Create fresh MemoryGraph instance (cold start)
        print("❄️  Creating fresh MemoryGraph instance (cold start)...")
        memory_graph = MemoryGraph()
        
        if memory_graph.mongo_client is None or memory_graph.db is None:
            print("❌ MongoDB not available")
            return False
        
        # Test 1: First API key lookup (cold)
        print("\n🔍 Testing first API key lookup (cold)...")
        cold_start = time.time()
        
        # Find a real API key first
        sample_user = memory_graph.db['_User'].find_one({'userAPIkey': {'$type': 'string'}})
        if not sample_user:
            print("❌ No API keys found for testing")
            return False
        
        api_key = sample_user['userAPIkey']
        
        # Now test the cold lookup
        cold_lookup_start = time.time()
        user = memory_graph.db['_User'].find_one({'userAPIkey': api_key})
        cold_lookup_time = (time.time() - cold_lookup_start) * 1000
        
        print(f"❄️  Cold API key lookup: {cold_lookup_time:.2f}ms")
        
        # Test 2: Second API key lookup (warm)
        print("\n🔥 Testing second API key lookup (warm)...")
        warm_lookup_start = time.time()
        user = memory_graph.db['_User'].find_one({'userAPIkey': api_key})
        warm_lookup_time = (time.time() - warm_lookup_start) * 1000
        
        print(f"🔥 Warm API key lookup: {warm_lookup_time:.2f}ms")
        
        # Test 3: Test with warmup
        print("\n🔥 Testing with manual warmup...")
        memory_graph_warmed = MemoryGraph()
        await memory_graph_warmed.warm_mongodb_connection()
        
        warmed_lookup_start = time.time()
        user = memory_graph_warmed.db['_User'].find_one({'userAPIkey': api_key})
        warmed_lookup_time = (time.time() - warmed_lookup_start) * 1000
        
        print(f"🔥 Warmed API key lookup: {warmed_lookup_time:.2f}ms")
        
        # Analysis
        print(f"\n📊 Performance Analysis:")
        print(f"Cold start: {cold_lookup_time:.2f}ms")
        print(f"Warm (same connection): {warm_lookup_time:.2f}ms")
        print(f"After warmup: {warmed_lookup_time:.2f}ms")
        
        improvement = cold_lookup_time - warmed_lookup_time
        improvement_pct = (improvement / cold_lookup_time) * 100
        
        print(f"Improvement: {improvement:.2f}ms ({improvement_pct:.1f}%)")
        
        if improvement > 50:  # 50ms improvement threshold
            print("✅ Warmup provides significant performance improvement")
            return True
        else:
            print("⚠️  Warmup improvement is marginal")
            return False
        
    except Exception as e:
        print(f"❌ Error during health check: {e}")
        return False


async def test_api_endpoint_performance():
    """Test actual API endpoint performance"""
    print(f"\n🌐 Testing API Endpoint Performance")
    print("-" * 50)
    
    try:
        # Get API URL from environment
        api_url = os.getenv('PYTHON_SERVER_URL', 'http://localhost:8000')
        
        # First, test if server is reachable
        print("🔍 Checking if server is running...")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{api_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        print("✅ Server is running")
                    else:
                        print(f"⚠️  Server returned status {response.status}")
            except (aiohttp.ClientError, asyncio.TimeoutError):
                print("❌ Server not reachable - skipping API endpoint test")
                print("💡 To test API performance, start the server with: python main.py")
                return None  # Return None to indicate test was skipped
        
        # Find a real API key for testing
        memory_graph = MemoryGraph()
        if memory_graph.mongo_client is None or memory_graph.db is None:
            print("❌ MongoDB not available")
            return False
        
        sample_user = memory_graph.db['_User'].find_one({'userAPIkey': {'$type': 'string'}})
        if not sample_user:
            print("❌ No API keys found for testing")
            return False
        
        api_key = sample_user['userAPIkey']
        
        # Test API endpoint
        async with aiohttp.ClientSession() as session:
            headers = {'X-API-Key': api_key}
            
            # Test 1: First API call
            print("🔍 Testing first API call...")
            start_time = time.time()
            async with session.get(f"{api_url}/v1/users/me", headers=headers) as response:
                first_api_time = (time.time() - start_time) * 1000
                if response.status == 200:
                    print(f"✅ First API call: {first_api_time:.2f}ms")
                else:
                    print(f"❌ First API call failed: {response.status}")
                    return False
            
            # Test 2: Second API call
            print("🔍 Testing second API call...")
            start_time = time.time()
            async with session.get(f"{api_url}/v1/users/me", headers=headers) as response:
                second_api_time = (time.time() - start_time) * 1000
                if response.status == 200:
                    print(f"✅ Second API call: {second_api_time:.2f}ms")
                else:
                    print(f"❌ Second API call failed: {response.status}")
                    return False
            
            # Analysis
            print(f"\n📊 API Performance Analysis:")
            print(f"First API call: {first_api_time:.2f}ms")
            print(f"Second API call: {second_api_time:.2f}ms")
            
            if first_api_time > 200:
                print("⚠️  First API call is slow - warmup may not be working")
                return False
            elif first_api_time < 100:
                print("✅ First API call is fast - warmup is working!")
                return True
            else:
                print("✅ First API call is acceptable")
                return True
                
    except Exception as e:
        print(f"❌ Error testing API endpoint: {e}")
        return False


async def main():
    """Main health check function"""
    print("🏥 MongoDB Health Check")
    print("=" * 50)
    
    # Test 1: Cold start performance
    cold_start_ok = await test_cold_start_performance()
    
    # Test 2: API endpoint performance (if available)
    api_ok = await test_api_endpoint_performance()
    
    # Final analysis
    print(f"\n🎯 Final Results:")
    print(f"Cold start optimization: {'✅ PASS' if cold_start_ok else '❌ FAIL'}")
    print(f"API endpoint performance: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if cold_start_ok and api_ok:
        print("\n🎉 MongoDB warmup is working correctly!")
        print("💡 Your API should have fast first-request performance")
        return True
    else:
        print("\n⚠️  MongoDB warmup needs attention")
        print("💡 Consider checking connection settings and warmup configuration")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 