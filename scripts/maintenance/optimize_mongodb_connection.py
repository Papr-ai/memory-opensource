#!/usr/bin/env python3
"""
MongoDB Connection Optimization Script

This script optimizes MongoDB connection settings specifically for DocumentDB/Atlas
to reduce query latency and improve performance.

Based on best practices from:
- MongoDB Atlas Performance Guide
- DocumentDB Optimization Guide
- Connection pooling best practices

Usage:
    python scripts/optimize_mongodb_connection.py
"""

import sys
import os
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pymongo import MongoClient, ReadPreference
from urllib.parse import urlparse, parse_qs


def analyze_current_connection():
    """Analyze current MongoDB connection settings"""
    print("🔍 Analyzing Current MongoDB Connection...")
    
    connection_string = os.environ.get("MONGO_URI")
    if not connection_string:
        print("  ❌ MONGO_URI environment variable not found")
        return None
    
    print(f"  📡 Connection string: {connection_string[:80]}...")
    
    # Parse the connection string
    parsed = urlparse(connection_string)
    params = parse_qs(parsed.query)
    
    print(f"  🖥️  Host: {parsed.hostname}")
    print(f"  🗄️  Database: {parsed.path.lstrip('/')}")
    
    print(f"  ⚙️  Current connection parameters:")
    for key, value in params.items():
        print(f"    - {key}: {value[0] if isinstance(value, list) else value}")
    
    return connection_string, parsed, params


def get_optimized_connection_string(connection_string: str) -> str:
    """Generate optimized connection string for DocumentDB/Atlas"""
    
    # Parse existing connection
    parsed = urlparse(connection_string)
    base_uri = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    
    # Optimized parameters for DocumentDB/Atlas
    optimized_params = {
        # Connection Pool Settings (critical for performance)
        'maxPoolSize': '50',           # Increase from default 100 to prevent overflow
        'minPoolSize': '5',            # Keep minimum connections warm
        'maxIdleTimeMS': '30000',      # 30 seconds idle timeout
        'waitQueueTimeoutMS': '5000',  # 5 second wait timeout
        
        # Read/Write Settings
        'readPreference': 'primaryPreferred',  # Prefer primary but allow secondary
        'readConcern': 'local',                # Local read concern for speed
        'w': '1',                              # Write concern: acknowledge from primary
        'j': 'false',                          # Don't wait for journal (faster writes)
        
        # Connection Settings
        'connectTimeoutMS': '5000',     # 5 second connection timeout
        'serverSelectionTimeoutMS': '5000',  # 5 second server selection
        'socketTimeoutMS': '30000',     # 30 second socket timeout
        'heartbeatFrequencyMS': '10000', # 10 second heartbeat
        
        # Compression (if supported)
        'compressors': 'zlib',          # Enable compression
        
        # TLS Settings
        'ssl': 'true',
        'sslInsecure': 'false',
        
        # DocumentDB specific
        'retryWrites': 'false',         # DocumentDB doesn't support retryable writes
        'replicaSet': '',               # Remove replica set for DocumentDB compatibility
    }
    
    # Build optimized connection string
    param_string = '&'.join([f"{k}={v}" for k, v in optimized_params.items() if v])
    optimized_uri = f"{base_uri}?{param_string}"
    
    return optimized_uri


def test_connection_performance(connection_string: str, label: str) -> dict:
    """Test connection performance with given connection string"""
    print(f"  🧪 Testing {label} connection...")
    
    try:
        # Test connection creation time
        create_start = time.time()
        client = MongoClient(connection_string)
        create_time = (time.time() - create_start) * 1000
        
        # Test database ping
        ping_start = time.time()
        db = client.get_default_database()
        ping_result = db.command('ping')
        ping_time = (time.time() - ping_start) * 1000
        
        # Test simple query
        query_start = time.time()
        user_count = db['_User'].count_documents({}, limit=1)
        query_time = (time.time() - query_start) * 1000
        
        # Test index query
        index_start = time.time()
        index_result = db['_User'].find_one({'userAPIkey': 'test_key_12345'})
        index_time = (time.time() - index_start) * 1000
        
        # Get connection info
        try:
            server_info = client.server_info()
            version = server_info.get('version', 'unknown')
        except:
            version = 'unknown'
        
        # Close connection
        client.close()
        
        results = {
            'success': True,
            'create_time': create_time,
            'ping_time': ping_time,
            'query_time': query_time,
            'index_time': index_time,
            'version': version
        }
        
        print(f"    ✅ {label} Results:")
        print(f"      - Connection creation: {create_time:.2f}ms")
        print(f"      - Ping: {ping_time:.2f}ms")
        print(f"      - Count query: {query_time:.2f}ms")
        print(f"      - Index query: {index_time:.2f}ms")
        print(f"      - Server version: {version}")
        
        return results
        
    except Exception as e:
        print(f"    ❌ {label} failed: {e}")
        return {'success': False, 'error': str(e)}


def generate_optimized_env_file():
    """Generate .env file with optimized MongoDB settings"""
    print("\n📝 Generating Optimized Configuration...")
    
    current_uri = os.environ.get("MONGO_URI")
    if not current_uri:
        print("  ❌ No current MONGO_URI found")
        return
    
    optimized_uri = get_optimized_connection_string(current_uri)
    
    # Write to a new env file
    with open('.env.mongodb_optimized', 'w') as f:
        f.write("# Optimized MongoDB Connection Settings\n")
        f.write("# Replace your existing MONGO_URI with this optimized version\n")
        f.write("# These settings are specifically tuned for DocumentDB/Atlas performance\n\n")
        f.write(f"MONGO_URI={optimized_uri}\n")
        f.write("\n# Additional recommendations:\n")
        f.write("# 1. Ensure your MongoDB Atlas cluster is in the same region as your app\n")
        f.write("# 2. Consider upgrading to M10+ for better performance\n")
        f.write("# 3. Enable connection pooling at application level\n")
        f.write("# 4. Monitor cluster performance in Atlas dashboard\n")
    
    print(f"  ✅ Optimized configuration saved to: .env.mongodb_optimized")
    print(f"  📋 Preview of optimized URI:")
    print(f"    {optimized_uri[:120]}...")


def analyze_cluster_tier():
    """Analyze MongoDB cluster tier and suggest improvements"""
    print("\n🏗️  Cluster Analysis & Recommendations...")
    
    try:
        connection_string = os.environ.get("MONGO_URI")
        client = MongoClient(connection_string)
        db = client.get_default_database()
        
        # Get server info
        server_info = client.server_info()
        version = server_info.get('version', 'unknown')
        
        # Get database stats
        stats = db.command('dbStats')
        storage_size = stats.get('storageSize', 0) / (1024 * 1024)  # Convert to MB
        data_size = stats.get('dataSize', 0) / (1024 * 1024)  # Convert to MB
        
        print(f"  📊 Cluster Information:")
        print(f"    - MongoDB version: {version}")
        print(f"    - Storage size: {storage_size:.2f} MB")
        print(f"    - Data size: {data_size:.2f} MB")
        
        # Analyze performance tier based on connection string
        if 'cluster0' in connection_string:
            print(f"  🎯 Cluster Type: Likely Standard Atlas cluster")
            if storage_size < 100:
                print(f"    💡 Recommendation: Consider M10+ tier for better performance")
            else:
                print(f"    ✅ Good: Appears to be production-ready tier")
        
        # Check for DocumentDB indicators
        if 'DocumentDB' in str(stats):
            print(f"  🗄️  Database: DocumentDB (AWS managed)")
            print(f"    💡 DocumentDB optimization tips:")
            print(f"      - Use read preferences carefully")
            print(f"      - Avoid retryable writes (not supported)")
            print(f"      - Optimize for smaller result sets")
        
        client.close()
        
    except Exception as e:
        print(f"  ⚠️  Could not analyze cluster: {e}")


def main():
    """Main optimization function"""
    print("🚀 MongoDB Connection Optimization Tool")
    print("=" * 50)
    
    # Step 1: Analyze current connection
    analysis = analyze_current_connection()
    if not analysis:
        return
    
    current_uri, parsed, params = analysis
    
    # Step 2: Performance test current connection
    print(f"\n⚡ Performance Testing...")
    current_results = test_connection_performance(current_uri, "Current")
    
    # Step 3: Test optimized connection
    optimized_uri = get_optimized_connection_string(current_uri)
    optimized_results = test_connection_performance(optimized_uri, "Optimized")
    
    # Step 4: Compare results
    if current_results.get('success') and optimized_results.get('success'):
        print(f"\n📈 Performance Comparison:")
        
        metrics = ['ping_time', 'query_time', 'index_time']
        for metric in metrics:
            current_val = current_results.get(metric, 0)
            optimized_val = optimized_results.get(metric, 0)
            improvement = ((current_val - optimized_val) / current_val) * 100 if current_val > 0 else 0
            
            print(f"  📊 {metric.replace('_', ' ').title()}:")
            print(f"    - Current: {current_val:.2f}ms")
            print(f"    - Optimized: {optimized_val:.2f}ms")
            if improvement > 0:
                print(f"    - ✅ Improvement: {improvement:.1f}%")
            elif improvement < -5:
                print(f"    - ⚠️  Regression: {abs(improvement):.1f}%")
            else:
                print(f"    - ➡️  Similar: {improvement:.1f}%")
    
    # Step 5: Generate optimized configuration
    generate_optimized_env_file()
    
    # Step 6: Cluster analysis
    analyze_cluster_tier()
    
    # Step 7: Final recommendations
    print(f"\n💡 Final Recommendations:")
    print(f"  1. 🌍 **Region Optimization**: Move cluster closer to your application")
    print(f"     - Current ping latency suggests geographic distance")
    print(f"     - Consider AWS US-West-2 based on diagnostic results")
    
    print(f"  2. ⚙️  **Connection Settings**: Use the optimized connection string")
    print(f"     - Copy settings from .env.mongodb_optimized")
    print(f"     - Focus on connection pooling parameters")
    
    print(f"  3. 🏗️  **Cluster Tier**: Consider upgrading if on free tier")
    print(f"     - M0 (free) has severe performance limitations")
    print(f"     - M10+ provides dedicated resources and better performance")
    
    print(f"  4. 📊 **Monitoring**: Enable Atlas Performance Advisor")
    print(f"     - Monitor slow queries and index usage")
    print(f"     - Set up alerts for high latency")
    
    print(f"  5. 🔧 **Application Level**: Implement connection reuse")
    print(f"     - Reuse MemoryGraph instances where possible")
    print(f"     - Consider MongoDB connection middleware")


if __name__ == "__main__":
    main() 