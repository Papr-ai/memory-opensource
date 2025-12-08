#!/usr/bin/env python3
"""
Debug script to check environment variables
"""
import os

# Load environment variables the same way the app does
from dotenv import load_dotenv, find_dotenv

# Check USE_DOTENV
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
print(f"USE_DOTENV: {use_dotenv}")

if use_dotenv:
    ENV_FILE = find_dotenv()
    print(f"Found .env file: {ENV_FILE}")
    if ENV_FILE:
        load_dotenv(ENV_FILE)
        print("Loaded .env file")
    else:
        print("No .env file found")
else:
    print("USE_DOTENV is false, not loading .env file")

# Check key environment variables
env_vars = [
    "MONGO_URI",
    "DATABASE_URI",
    "PARSE_SERVER_URL",
    "PARSE_APPLICATION_ID",
    "PARSE_MASTER_KEY",
    "MONGODB_URI"
]

print("\nEnvironment Variables:")
for var in env_vars:
    value = os.getenv(var)
    if value:
        print(f"✅ {var}: {value[:50]}..." if len(value) > 50 else f"✅ {var}: {value}")
    else:
        print(f"❌ {var}: Not set")

# Test MongoDB connection
print(f"\nMongoDB connection logic debug:")
mongo_uri = os.getenv("MONGO_URI")
database_uri = os.getenv("DATABASE_URI")

print(f"MONGO_URI: {'Set' if mongo_uri else 'Not set'}")
print(f"DATABASE_URI: {'Set' if database_uri else 'Not set'}")

if database_uri:
    print(f"DATABASE_URI contains 'mongodb': {'Yes' if 'mongodb' in database_uri else 'No'}")

from services.mongo_client import get_mongo_db
print(f"\nMongoDB connection test:")
try:
    db = get_mongo_db()
    if db:
        print(f"✅ MongoDB connected: {db.name}")
    else:
        print("❌ MongoDB not available (returned None)")
        print("This means MONGO_URI is not set AND DATABASE_URI is either not set or doesn't contain 'mongodb'")
except Exception as e:
    print(f"❌ MongoDB error: {e}")

print(f"\nSuggestion: Set DATABASE_URI in your ~/.zshrc or .env file to your MongoDB connection string")
print("Example: export DATABASE_URI='mongodb://localhost:27017/parsedev'")
