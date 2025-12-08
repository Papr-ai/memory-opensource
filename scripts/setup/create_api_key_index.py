#!/usr/bin/env python3
"""
MongoDB Index Creation Script for API Key Performance Optimization

This script creates an index on the userAPIkey field in the _User collection
to dramatically improve API key lookup performance.

Usage:
    python scripts/create_api_key_index.py

The script will:
1. Connect to MongoDB using the current environment configuration
2. Create a unique index on the userAPIkey field
3. Report timing and success/failure status
4. Show all existing indexes for verification
"""

import sys
import os
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from memory.memory_graph import MemoryGraph


def create_api_key_index():
    """Create MongoDB index on userAPIkey field for fast lookups"""
    print("🚀 MongoDB API Key Index Creation Script")
    print("=" * 50)
    
    try:
        # Initialize connection to MongoDB
        print("📡 Connecting to MongoDB...")
        memory_graph = MemoryGraph()
        db = memory_graph.db
        
        if db is None:
            print("❌ Failed to connect to MongoDB database")
            return False
            
        print(f"✅ Connected to MongoDB database: {db.name}")
        
        # Check if index already exists
        print("\n🔍 Checking existing indexes...")
        existing_indexes = list(db['_User'].list_indexes())
        
        api_key_indexes = []
        for idx in existing_indexes:
            # Check both the key field and index name for userAPIkey
            key_str = str(idx.get('key', {}))
            name_str = str(idx.get('name', ''))
            if 'userAPIkey' in key_str or 'userAPIkey' in name_str:
                api_key_indexes.append(idx)
        
        if api_key_indexes:
            print("ℹ️  API key index already exists:")
            for idx in api_key_indexes:
                print(f"   - {idx['name']}: {idx.get('key', {})}")
            print("✅ No action needed - index already optimized!")
            return True
        
        # Check for null userAPIkey values
        print("\n🔍 Checking for null userAPIkey values...")
        null_count = db['_User'].count_documents({'userAPIkey': None})
        missing_count = db['_User'].count_documents({'userAPIkey': {'$exists': False}})
        total_users = db['_User'].count_documents({})
        
        print(f"📊 Users with null userAPIkey: {null_count}")
        print(f"📊 Users with missing userAPIkey: {missing_count}")
        print(f"📊 Total users: {total_users}")
        
        if null_count > 0 or missing_count > 0:
            print("⚠️  Found users with null/missing userAPIkey values")
            print("💡 Using partial index to handle null values gracefully")
        
        # Create the indexes for optimal performance
        print("\n⚡ Creating indexes for optimal MongoDB performance...")
        start_time = time.time()
        
        # 1. Primary API key index (most important)
        # Use partial index compatible with DocumentDB - only index documents where userAPIkey exists and is a string
        result1 = db['_User'].create_index(
            'userAPIkey', 
            unique=True,     # API keys should be unique
            background=True, # Create in background to avoid blocking
            name='userAPIkey_1',  # Explicit name for easier management
            partialFilterExpression={'userAPIkey': {'$type': 'string'}}
        )
        print(f"✅ userAPIkey index: {result1}")
        
        # 2. Index on workspace_follower._id for fast lookups
        try:
            result2 = db['workspace_follower'].create_index(
                '_id',
                background=True,
                name='workspace_follower_id_1'
            )
            print(f"✅ workspace_follower._id index: {result2}")
        except Exception as e:
            print(f"ℹ️  workspace_follower._id index already exists: {e}")
        
        # 3. Index on workspace_follower._p_workspace for pointer lookups
        try:
            result3 = db['workspace_follower'].create_index(
                '_p_workspace',
                background=True,
                name='workspace_follower_workspace_ptr_1'
            )
            print(f"✅ workspace_follower._p_workspace index: {result3}")
        except Exception as e:
            print(f"ℹ️  workspace_follower._p_workspace index already exists: {e}")
        
        index_time = (time.time() - start_time) * 1000
        print(f"⚡ All index creation took: {index_time:.2f}ms")
        
        # Verify the index was created
        print("\n📋 Verifying index creation...")
        updated_indexes = list(db['_User'].list_indexes())
        
        print("Current indexes on _User collection:")
        for idx in updated_indexes:
            key_info = idx.get('key', {})
            unique_info = " (UNIQUE)" if idx.get('unique', False) else ""
            print(f"  - {idx['name']}: {key_info}{unique_info}")
        
        # Performance test
        print("\n🏃 Testing API key lookup performance...")
        test_start = time.time()
        
        # Find a user with an API key for testing
        sample_user = db['_User'].find_one({"userAPIkey": {"$exists": True, "$ne": None}})
        
        if sample_user and 'userAPIkey' in sample_user:
            api_key = sample_user['userAPIkey']
            
            # Test the lookup speed
            lookup_start = time.time()
            result = db['_User'].find_one({"userAPIkey": api_key})
            lookup_time = (time.time() - lookup_start) * 1000
            
            if result:
                print(f"✅ Test lookup successful in {lookup_time:.2f}ms")
                print(f"   Found user: {result.get('username', 'N/A')} ({result['_id']})")
            else:
                print("⚠️  Test lookup failed - no user found")
        else:
            print("ℹ️  No users with API keys found for performance testing")
        
        print(f"\n🎉 Index creation completed successfully!")
        print("🚀 API key lookups should now be significantly faster!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Try to provide more context
        try:
            indexes = list(db['_User'].list_indexes())
            print(f"\n📋 Current indexes (for debugging):")
            for idx in indexes:
                print(f"  - {idx['name']}: {idx.get('key', {})}")
        except Exception as e2:
            print(f"❌ Error checking existing indexes: {e2}")
        
        return False


def recreate_regular_index():
    """Recreate the userAPIkey index as a regular unique index (not partial)"""
    print("🔄 Recreating userAPIkey index as regular unique index")
    print("=" * 50)
    
    try:
        # Initialize connection to MongoDB
        print("📡 Connecting to MongoDB...")
        memory_graph = MemoryGraph()
        db = memory_graph.db
        
        if db is None:
            print("❌ Failed to connect to MongoDB database")
            return False
            
        print(f"✅ Connected to MongoDB database: {db.name}")
        
        # Check if all users have API keys
        print("\n🔍 Checking if all users have API keys...")
        total_users = db['_User'].count_documents({})
        users_with_keys = db['_User'].count_documents({'userAPIkey': {'$type': 'string'}})
        
        print(f"📊 Total users: {total_users}")
        print(f"✅ Users with API keys: {users_with_keys}")
        
        if users_with_keys != total_users:
            print(f"⚠️  Not all users have API keys ({users_with_keys}/{total_users})")
            print(f"💡 Run the generate_missing_api_keys.py script first")
            return False
        
        print("✅ All users have API keys - proceeding with regular index creation")
        
        # Drop the existing partial index
        print("\n🗑️  Dropping existing partial index...")
        try:
            db['_User'].drop_index('userAPIkey_1')
            print("✅ Dropped existing userAPIkey_1 index")
        except Exception as e:
            print(f"ℹ️  Could not drop existing index: {e}")
        
        # Create a regular unique index
        print("\n⚡ Creating regular unique index...")
        start_time = time.time()
        
        result = db['_User'].create_index(
            'userAPIkey',
            unique=True,
            background=True,
            name='userAPIkey_1'
        )
        
        index_time = (time.time() - start_time) * 1000
        print(f"✅ Regular unique index created: {result}")
        print(f"⚡ Index creation took: {index_time:.2f}ms")
        
        # Verify the index
        print("\n📋 Verifying new index...")
        updated_indexes = list(db['_User'].list_indexes())
        
        for idx in updated_indexes:
            if idx['name'] == 'userAPIkey_1':
                key_info = idx.get('key', {})
                unique_info = " (UNIQUE)" if idx.get('unique', False) else ""
                partial_info = " (PARTIAL)" if 'partialFilterExpression' in idx else ""
                print(f"  ✅ {idx['name']}: {key_info}{unique_info}{partial_info}")
                
                if 'partialFilterExpression' not in idx:
                    print("  🎉 Index is now a regular unique index (not partial)")
                else:
                    print("  ⚠️  Index is still partial")
        
        # Test performance with real API key
        print("\n🏃 Testing index performance...")
        sample_user = db['_User'].find_one({"userAPIkey": {"$type": "string"}})
        
        if sample_user:
            test_start = time.time()
            result = db['_User'].find_one({"userAPIkey": sample_user['userAPIkey']})
            test_time = (time.time() - test_start) * 1000
            print(f"✅ Index lookup test: {test_time:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Error recreating index: {e}")
        return False


def main():
    """Main entry point"""
    print(f"Environment: {os.getenv('NODE_ENV', 'development')}")
    print(f"MongoDB URI: {os.getenv('MONGO_URI', 'Not set')}...\n")
    
    # Check if user wants to recreate as regular index
    if len(sys.argv) > 1 and sys.argv[1] == '--regular':
        success = recreate_regular_index()
    else:
        success = create_api_key_index()
    
    if success:
        print("\n✅ Script completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Script failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 