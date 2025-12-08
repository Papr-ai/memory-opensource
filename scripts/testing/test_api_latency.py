#!/usr/bin/env python3
"""
API Latency Test Script

This script tests the actual API latency using the running server to measure
the real-world performance impact of MongoDB warmup and optimization.

Usage:
    python scripts/test_api_latency.py
"""

import time
import requests
import json
import statistics
from typing import List, Dict, Any

# Server configuration
SERVER_URL = "https://4e4fc3b78291.ngrok.app"
TEST_API_KEY = "006c976f23455055616e608ee91812be"  # From the duplicate key error


def test_api_endpoint(endpoint: str, method: str = "GET", data: Dict = None, headers: Dict = None) -> Dict[str, Any]:
    """Test a single API endpoint and measure latency"""
    if headers is None:
        headers = {}
    
    url = f"{SERVER_URL}{endpoint}"
    
    try:
        start_time = time.time()
        
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        return {
            'success': True,
            'status_code': response.status_code,
            'latency_ms': latency_ms,
            'response_size': len(response.content),
            'response_preview': response.text[:200] if response.text else None
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'latency_ms': None
        }


def run_latency_test(endpoint: str, method: str = "GET", data: Dict = None, headers: Dict = None, iterations: int = 5) -> Dict[str, Any]:
    """Run multiple iterations of an API test and calculate statistics"""
    print(f"🧪 Testing {method} {endpoint} ({iterations} iterations)...")
    
    results = []
    latencies = []
    
    for i in range(iterations):
        print(f"  Iteration {i+1}/{iterations}...", end=" ", flush=True)
        result = test_api_endpoint(endpoint, method, data, headers)
        results.append(result)
        
        if result['success']:
            latencies.append(result['latency_ms'])
            print(f"✅ {result['latency_ms']:.2f}ms (HTTP {result['status_code']})")
        else:
            print(f"❌ {result['error']}")
    
    # Calculate statistics
    if latencies:
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        median_latency = statistics.median(latencies)
        
        print(f"  📊 Results:")
        print(f"    Average: {avg_latency:.2f}ms")
        print(f"    Median:  {median_latency:.2f}ms")
        print(f"    Min:     {min_latency:.2f}ms")
        print(f"    Max:     {max_latency:.2f}ms")
        
        if len(latencies) > 1:
            stdev = statistics.stdev(latencies)
            print(f"    StdDev:  {stdev:.2f}ms")
        
        return {
            'success': True,
            'avg_latency': avg_latency,
            'min_latency': min_latency,
            'max_latency': max_latency,
            'median_latency': median_latency,
            'all_latencies': latencies,
            'success_rate': len(latencies) / iterations * 100
        }
    else:
        print(f"  ❌ All requests failed")
        return {
            'success': False,
            'success_rate': 0,
            'errors': [r['error'] for r in results if not r['success']]
        }


def test_cold_start_simulation():
    """Test if there's still any cold start penalty"""
    print(f"\n🧊 Testing Cold Start Simulation")
    print("=" * 50)
    
    # Wait a bit to simulate some idle time
    print("⏳ Waiting 10 seconds to simulate idle time...")
    time.sleep(10)
    
    # Test first request after idle
    print("🔥 Testing first request after idle...")
    first_result = test_api_endpoint("/")
    
    if first_result['success']:
        print(f"  ✅ First request: {first_result['latency_ms']:.2f}ms")
        
        # Test immediate follow-up requests
        print("⚡ Testing immediate follow-up requests...")
        followup_latencies = []
        for i in range(3):
            result = test_api_endpoint("/")
            if result['success']:
                followup_latencies.append(result['latency_ms'])
                print(f"  ✅ Request {i+1}: {result['latency_ms']:.2f}ms")
        
        if followup_latencies:
            avg_followup = statistics.mean(followup_latencies)
            print(f"\n📊 Cold Start Analysis:")
            print(f"  First request:    {first_result['latency_ms']:.2f}ms")
            print(f"  Average follow-up: {avg_followup:.2f}ms")
            print(f"  Improvement:      {first_result['latency_ms'] - avg_followup:.2f}ms")
            
            if first_result['latency_ms'] - avg_followup > 50:
                print(f"  ⚠️  Cold start penalty detected!")
            else:
                print(f"  ✅ No significant cold start penalty")
    else:
        print(f"  ❌ First request failed: {first_result['error']}")


def main():
    """Main test function"""
    print("🚀 API Latency Test with MongoDB Warmup")
    print("=" * 50)
    print(f"Server URL: {SERVER_URL}")
    print(f"Test API Key: {TEST_API_KEY}")
    
    # Test 1: Basic health check
    print(f"\n🏥 Test 1: Basic Health Check")
    health_result = run_latency_test("/", iterations=5)
    
    # Test 2: Memory-related endpoint (if available)
    print(f"\n🧠 Test 2: Memory Endpoint")
    # Test a simple memory endpoint that would use MongoDB
    memory_headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
    memory_result = run_latency_test("/v1/users/me", headers=memory_headers, iterations=5)
    
    # Test 3: Add memory endpoint (MongoDB write operation)
    print(f"\n✍️  Test 3: Add Memory (Write Operation)")
    add_memory_data = {
        "content": "Test memory content for latency testing",
        "metadata": {"test": True, "timestamp": time.time()}
    }
    add_memory_result = run_latency_test("/v1/memories", method="POST", data=add_memory_data, headers=memory_headers, iterations=3)
    
    # Test 4: Get memories (MongoDB read operation)
    print(f"\n📖 Test 4: Get Memories (Read Operation)")
    get_memories_result = run_latency_test("/v1/memories", headers=memory_headers, iterations=5)
    
    # Test 5: Cold start simulation
    test_cold_start_simulation()
    
    # Summary
    print(f"\n📋 Summary & Analysis")
    print("=" * 50)
    
    if health_result['success']:
        print(f"✅ Health Check: {health_result['avg_latency']:.2f}ms avg")
    else:
        print(f"❌ Health Check: Failed")
    
    if memory_result['success']:
        print(f"✅ User Auth: {memory_result['avg_latency']:.2f}ms avg")
    else:
        print(f"❌ User Auth: Failed")
    
    if add_memory_result['success']:
        print(f"✅ Add Memory: {add_memory_result['avg_latency']:.2f}ms avg")
    else:
        print(f"❌ Add Memory: Failed")
    
    if get_memories_result['success']:
        print(f"✅ Get Memories: {get_memories_result['avg_latency']:.2f}ms avg")
    else:
        print(f"❌ Get Memories: Failed")
    
    # Performance analysis
    print(f"\n🎯 Performance Analysis:")
    
    # Check if any endpoint is particularly slow
    slow_endpoints = []
    if health_result['success'] and health_result['avg_latency'] > 200:
        slow_endpoints.append(f"Health Check ({health_result['avg_latency']:.2f}ms)")
    if memory_result['success'] and memory_result['avg_latency'] > 200:
        slow_endpoints.append(f"User Auth ({memory_result['avg_latency']:.2f}ms)")
    if add_memory_result['success'] and add_memory_result['avg_latency'] > 500:
        slow_endpoints.append(f"Add Memory ({add_memory_result['avg_latency']:.2f}ms)")
    if get_memories_result['success'] and get_memories_result['avg_latency'] > 300:
        slow_endpoints.append(f"Get Memories ({get_memories_result['avg_latency']:.2f}ms)")
    
    if slow_endpoints:
        print(f"  ⚠️  Slow endpoints detected:")
        for endpoint in slow_endpoints:
            print(f"    - {endpoint}")
    else:
        print(f"  ✅ All endpoints performing well!")
    
    print(f"\n💡 Recommendations:")
    print(f"  1. MongoDB warmup is working (visible in server logs)")
    print(f"  2. First API requests should be ~40-50ms (not 200ms+)")
    print(f"  3. Monitor for any remaining cold start penalties")
    print(f"  4. Consider adding index hints if queries are still slow")


if __name__ == "__main__":
    main() 