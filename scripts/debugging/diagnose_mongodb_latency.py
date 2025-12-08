#!/usr/bin/env python3
"""
MongoDB Atlas Latency Diagnostic Script

This script performs comprehensive latency testing to identify why MongoDB queries are slow:
1. Network ping tests to Atlas regions
2. MongoDB connection latency tests
3. Simple query performance tests
4. Connection pool analysis
5. Index usage verification

Usage:
    python scripts/diagnose_mongodb_latency.py
"""

import sys
import os
import time
import asyncio
import subprocess
import socket
from urllib.parse import urlparse

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from memory.memory_graph import MemoryGraph


def ping_host(hostname: str, count: int = 4) -> dict:
    """Ping a hostname and return latency statistics"""
    try:
        print(f"🌐 Pinging {hostname}...")
        
        # Use ping command (works on macOS/Linux)
        result = subprocess.run(
            ['ping', '-c', str(count), hostname],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            
            # Extract latency from ping output
            latencies = []
            for line in lines:
                if 'time=' in line:
                    try:
                        time_part = line.split('time=')[1].split(' ')[0]
                        latency = float(time_part)
                        latencies.append(latency)
                    except (IndexError, ValueError):
                        continue
            
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                
                print(f"  ✅ Average: {avg_latency:.2f}ms")
                print(f"  ⚡ Min: {min_latency:.2f}ms")
                print(f"  🔥 Max: {max_latency:.2f}ms")
                
                return {
                    'success': True,
                    'avg': avg_latency,
                    'min': min_latency,
                    'max': max_latency,
                    'latencies': latencies
                }
        
        print(f"  ❌ Ping failed: {result.stderr}")
        return {'success': False, 'error': result.stderr}
        
    except Exception as e:
        print(f"  ❌ Error pinging {hostname}: {e}")
        return {'success': False, 'error': str(e)}


def test_tcp_connection(hostname: str, port: int) -> dict:
    """Test TCP connection latency to MongoDB"""
    try:
        print(f"🔌 Testing TCP connection to {hostname}:{port}...")
        
        start_time = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        
        result = sock.connect_ex((hostname, port))
        connect_time = (time.time() - start_time) * 1000
        
        sock.close()
        
        if result == 0:
            print(f"  ✅ TCP connection successful: {connect_time:.2f}ms")
            return {'success': True, 'latency': connect_time}
        else:
            print(f"  ❌ TCP connection failed: {result}")
            return {'success': False, 'error': f"Connection failed: {result}"}
            
    except Exception as e:
        print(f"  ❌ Error testing TCP connection: {e}")
        return {'success': False, 'error': str(e)}


async def test_mongodb_connection_latency():
    """Test MongoDB connection and basic query latency"""
    try:
        print(f"\n📊 Testing MongoDB Connection Latency...")
        
        # Initialize MongoDB connection
        memory_graph = MemoryGraph()
        
        if not memory_graph.mongo_client:
            print("  ❌ MongoDB client not available")
            return {'success': False, 'error': 'No MongoDB client'}
        
        db = memory_graph.db
        
        # Test 1: Simple ping command
        print(f"  🏓 Testing MongoDB ping...")
        ping_start = time.time()
        ping_result = db.command('ping')
        ping_time = (time.time() - ping_start) * 1000
        print(f"    ✅ MongoDB ping: {ping_time:.2f}ms")
        
        # Test 2: Count documents in _User collection
        print(f"  🔢 Testing simple count query...")
        count_start = time.time()
        user_count = db['_User'].count_documents({})
        count_time = (time.time() - count_start) * 1000
        print(f"    ✅ Count query: {count_time:.2f}ms (found {user_count} users)")
        
        # Test 3: Simple find_one query
        print(f"  🔍 Testing simple find_one query...")
        find_start = time.time()
        one_user = db['_User'].find_one({}, {'_id': 1})
        find_time = (time.time() - find_start) * 1000
        print(f"    ✅ Find one query: {find_time:.2f}ms")
        
        # Test 4: Index-based query (if userAPIkey index exists)
        print(f"  🔑 Testing API key index query...")
        api_key_start = time.time()
        # Use a dummy API key that won't exist
        api_user = db['_User'].find_one({'userAPIkey': 'nonexistent_test_key_12345'})
        api_key_time = (time.time() - api_key_start) * 1000
        print(f"    ✅ API key query: {api_key_time:.2f}ms (result: {'found' if api_user else 'not found'})")
        
        # Test 5: Connection pool stats
        print(f"  🏊 MongoDB Connection Pool Stats:")
        try:
            client_info = memory_graph.mongo_client.topology_description
            print(f"    📡 Topology type: {client_info.topology_type}")
            print(f"    🖥️  Server descriptions: {len(client_info.server_descriptions())}")
            for server in client_info.server_descriptions():
                print(f"      - {server}: {client_info.server_descriptions()[server].server_type}")
        except Exception as pool_e:
            print(f"    ⚠️  Could not get pool stats: {pool_e}")
        
        return {
            'success': True,
            'ping_ms': ping_time,
            'count_ms': count_time,
            'find_ms': find_time,
            'api_key_ms': api_key_time
        }
        
    except Exception as e:
        print(f"  ❌ Error testing MongoDB connection: {e}")
        return {'success': False, 'error': str(e)}


def get_mongodb_hostname_from_connection():
    """Extract MongoDB hostname from connection string"""
    try:
        import os
        connection_string = os.environ.get("MONGO_URI")
        
        if not connection_string:
            print("  ❌ MONGO_URI environment variable not found")
            return None
            
        print(f"  📡 Connection string: {connection_string[:50]}...")
            
        # Parse the connection string
        # Format: mongodb+srv://username:password@cluster.mongodb.net/database
        if 'mongodb+srv://' in connection_string:
            # Extract hostname from mongodb+srv connection
            parsed = urlparse(connection_string)
            hostname = parsed.hostname
            return hostname
        elif 'mongodb://' in connection_string:
            # Extract hostname from regular mongodb connection
            parsed = urlparse(connection_string)
            hostname = parsed.hostname
            port = parsed.port or 27017
            return hostname, port
            
        return None
        
    except Exception as e:
        print(f"❌ Error parsing MongoDB connection string: {e}")
        return None


async def test_index_performance():
    """Test index usage and performance"""
    try:
        print(f"\n📈 Testing Index Performance...")
        
        memory_graph = MemoryGraph()
        if not memory_graph.mongo_client:
            print("  ❌ MongoDB client not available")
            return
        
        db = memory_graph.db
        
        # Get all indexes on _User collection
        print(f"  📋 Current indexes on _User collection:")
        indexes = list(db['_User'].list_indexes())
        for idx in indexes:
            name = idx.get('name', 'unknown')
            key = idx.get('key', {})
            print(f"    - {name}: {dict(key)}")
        
        # Test query with explain - use a real API key for accurate testing
        print(f"  🔍 Testing query execution plan...")
        explain_start = time.time()
        
        # First, find a real API key to test with
        sample_user = db['_User'].find_one({"userAPIkey": {"$type": "string"}})
        
        if sample_user and 'userAPIkey' in sample_user:
            test_api_key = sample_user['userAPIkey']
            print(f"    🔑 Testing with real API key: {test_api_key[:10]}...")
            explain_result = db['_User'].find({'userAPIkey': test_api_key}).explain()
        else:
            print(f"    ⚠️  No real API keys found, testing with dummy key...")
        explain_result = db['_User'].find({'userAPIkey': 'test_key'}).explain()
        
        explain_time = (time.time() - explain_start) * 1000
        print(f"    ⚡ Explain query took: {explain_time:.2f}ms")
        
        # Check if index was used
        execution_stats = explain_result.get('executionStats', {})
        if execution_stats:
            print(f"    📊 Execution Stats:")
            print(f"      - Execution time: {execution_stats.get('executionTimeMillis', 'N/A')}ms")
            print(f"      - Total docs examined: {execution_stats.get('totalDocsExamined', 'N/A')}")
            print(f"      - Total keys examined: {execution_stats.get('totalKeysExamined', 'N/A')}")
            
            winning_plan = explain_result.get('queryPlanner', {}).get('winningPlan', {})
            if winning_plan:
                stage = winning_plan.get('stage', 'unknown')
                print(f"      - Winning plan stage: {stage}")
                
                if 'inputStage' in winning_plan:
                    input_stage = winning_plan['inputStage']
                    if input_stage.get('stage') == 'IXSCAN':
                        index_name = input_stage.get('indexName', 'unknown')
                        print(f"      ✅ Using index: {index_name}")
                    else:
                        print(f"      ⚠️  Not using index scan: {input_stage.get('stage')}")
        
    except Exception as e:
        print(f"  ❌ Error testing index performance: {e}")


async def main():
    """Main diagnostic function"""
    print("🚀 MongoDB Atlas Latency Diagnostic Tool")
    print("=" * 50)
    
    # Step 1: Get MongoDB connection info
    print(f"\n📡 Step 1: Connection Information")
    hostname_info = get_mongodb_hostname_from_connection()
    
    if hostname_info:
        if isinstance(hostname_info, tuple):
            hostname, port = hostname_info
            print(f"  🎯 MongoDB Host: {hostname}:{port}")
        else:
            hostname = hostname_info
            port = 27017  # Default MongoDB port
            print(f"  🎯 MongoDB Host: {hostname} (SRV record)")
    else:
        print(f"  ❌ Could not extract MongoDB hostname")
        return
    
    # Step 2: Network latency tests
    print(f"\n🌐 Step 2: Network Latency Tests")
    ping_results = {}
    
    # Ping the main hostname
    if hostname:
        ping_results['main'] = ping_host(hostname)
    
    # Test common AWS regions (where MongoDB Atlas typically runs)
    atlas_regions = [
        'cluster0-shard-00-00.mongodb.net',  # Common Atlas naming
        'ec2.us-east-1.amazonaws.com',       # AWS US East
        'ec2.us-west-2.amazonaws.com',       # AWS US West
        'ec2.eu-west-1.amazonaws.com',       # AWS Europe
    ]
    
    for region in atlas_regions:
        if region not in hostname:  # Don't ping the same host twice
            ping_results[region] = ping_host(region, count=2)
    
    # Step 3: TCP connection test
    print(f"\n🔌 Step 3: TCP Connection Test")
    if hostname:
        tcp_result = test_tcp_connection(hostname, port)
    
    # Step 4: MongoDB-specific tests
    print(f"\n📊 Step 4: MongoDB Performance Tests")
    mongodb_result = await test_mongodb_connection_latency()
    
    # Step 5: Index performance
    await test_index_performance()
    
    # Step 6: Summary and recommendations
    print(f"\n📋 Step 6: Summary & Recommendations")
    print("=" * 50)
    
    # Analyze ping results
    if ping_results.get('main', {}).get('success'):
        main_latency = ping_results['main']['avg']
        print(f"🌐 Network latency to MongoDB: {main_latency:.2f}ms")
        
        if main_latency > 100:
            print(f"  ⚠️  HIGH LATENCY DETECTED! Network latency > 100ms")
            print(f"  💡 Consider moving to a closer MongoDB Atlas region")
        elif main_latency > 50:
            print(f"  ⚠️  Moderate latency. Consider optimization")
        else:
            print(f"  ✅ Good network latency")
    
    # Analyze MongoDB results
    if mongodb_result.get('success'):
        ping_ms = mongodb_result.get('ping_ms', 0)
        find_ms = mongodb_result.get('find_ms', 0)
        api_key_ms = mongodb_result.get('api_key_ms', 0)
        
        print(f"📊 MongoDB query performance:")
        print(f"  - Ping: {ping_ms:.2f}ms")
        print(f"  - Simple find: {find_ms:.2f}ms")  
        print(f"  - API key query: {api_key_ms:.2f}ms")
        
        if api_key_ms > 200:
            print(f"  🚨 VERY SLOW API key queries detected!")
            print(f"  💡 Recommendations:")
            print(f"    1. Verify userAPIkey index exists and is being used")
            print(f"    2. Check MongoDB Atlas cluster tier (M0/M2/M5 have performance limits)")
            print(f"    3. Consider upgrading to higher tier cluster")
            print(f"    4. Check if cluster is in same region as your application")
        elif api_key_ms > 50:
            print(f"  ⚠️  Slow API key queries. Consider optimization")
        else:
            print(f"  ✅ Good MongoDB query performance")
    
    # General recommendations
    print(f"\n💡 General Recommendations:")
    print(f"  1. Ensure MongoDB Atlas cluster is in same region as your app")
    print(f"  2. Upgrade from free tier (M0) if currently using it")  
    print(f"  3. Consider connection pooling optimization")
    print(f"  4. Add MongoDB Atlas Performance Advisor monitoring")
    print(f"  5. Check cluster CPU/memory usage in Atlas dashboard")


if __name__ == "__main__":
    asyncio.run(main()) 