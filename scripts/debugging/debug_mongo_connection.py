#!/usr/bin/env python3
"""
Debug script to check MongoDB connection details
"""
import os
from urllib.parse import urlparse
from pymongo import MongoClient

# Load environment variables conditionally
from dotenv import load_dotenv
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    load_dotenv()

def debug_mongo_connection():
    print("🔍 MongoDB Connection Debug")
    print("=" * 50)
    
    # Check environment variables
    mongo_uri = os.getenv("MONGO_URI")
    print(f"MONGO_URI from env: {mongo_uri[:50]}..." if mongo_uri else "MONGO_URI: Not set")
    
    if mongo_uri:
        # Parse the URI to extract database name
        parsed = urlparse(mongo_uri)
        db_from_uri = parsed.path.lstrip('/')
        print(f"Database from URI path: '{db_from_uri}'")
        
        # Test connection
        try:
            client = MongoClient(mongo_uri)
            db = client.get_default_database()
            print(f"✅ Connected successfully")
            print(f"📊 Database name from client: '{db.name}'")
            
            # List collections to verify we're in the right place
            collections = db.list_collection_names()
            print(f"📂 Collections found: {collections[:5]}..." if len(collections) > 5 else f"📂 Collections found: {collections}")
            
            # Check if _User collection exists and has data
            if "_User" in collections:
                user_count = db["_User"].count_documents({})
                print(f"👥 _User collection has {user_count} documents")
                
                # Try to find a user with an API key
                sample_user = db["_User"].find_one({"userAPIkey": {"$exists": True, "$ne": None}})
                if sample_user:
                    api_key = sample_user.get("userAPIkey", "")
                    print(f"🔑 Found sample API key: {api_key[:10]}...")
                else:
                    print("❌ No users with API keys found")
            else:
                print("❌ _User collection not found")
                
            client.close()
        except Exception as e:
            print(f"❌ Connection failed: {e}")
    
    # Also check Parse Server URL
    parse_url = os.getenv("PARSE_SERVER_URL")
    print(f"\n🌐 Parse Server URL: {parse_url}")
    
    memory_url = os.getenv("PYTHON_SERVER_URL")
    print(f"🧠 Memory Server URL: {memory_url}")

if __name__ == "__main__":
    debug_mongo_connection() 