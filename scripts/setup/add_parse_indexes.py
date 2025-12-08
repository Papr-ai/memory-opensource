import os
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, OperationFailure

MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise Exception("MONGO_URI not set in environment")

client = MongoClient(MONGO_URI)
db = client.get_default_database()

print("Checking existing indexes...")

# Check existing indexes for all relevant collections
user_indexes = list(db["_User"].list_indexes())
session_indexes = list(db["_Session"].list_indexes())
role_indexes = list(db["_Role"].list_indexes())
join_users_role_indexes = list(db["_Join:users:_Role"].list_indexes())
workspace_follower_indexes = list(db["workspace_follower"].list_indexes())

print(f"Existing _User indexes: {[idx['name'] for idx in user_indexes]}")
print(f"Existing _Session indexes: {[idx['name'] for idx in session_indexes]}")
print(f"Existing _Role indexes: {[idx['name'] for idx in role_indexes]}")
print(f"Existing _Join:users:_Role indexes: {[idx['name'] for idx in join_users_role_indexes]}")
print(f"Existing workspace_follower indexes: {[idx['name'] for idx in workspace_follower_indexes]}")

# Index for API Key lookup
print("\nCreating index on _User.userAPIkey...")
try:
    # Check if index already exists
    existing_user_api_key_index = any(idx.get('key', {}).get('userAPIkey') for idx in user_indexes)
    if existing_user_api_key_index:
        print("Index on _User.userAPIkey already exists, skipping...")
    else:
        db["_User"].create_index("userAPIkey", unique=False, background=True)
        print("✓ Index on _User.userAPIkey created successfully")
except Exception as e:
    print(f"Error creating _User.userAPIkey index: {e}")

# Index for Session token lookup - handle null values
print("\nCreating index on _Session.sessionToken...")
try:
    # Check if index already exists
    existing_session_token_index = any(idx.get('key', {}).get('sessionToken') for idx in session_indexes)
    if existing_session_token_index:
        print("Index on _Session.sessionToken already exists, skipping...")
    else:
        # First, clean up null sessionToken values
        print("Cleaning up null sessionToken values...")
        result = db["_Session"].update_many(
            {"sessionToken": None},
            {"$unset": {"sessionToken": ""}}
        )
        print(f"Removed {result.modified_count} documents with null sessionToken")
        
        # Create sparse index to exclude null values
        db["_Session"].create_index("sessionToken", unique=True, background=True, sparse=True)
        print("✓ Index on _Session.sessionToken created successfully (sparse)")
except Exception as e:
    print(f"Error creating _Session.sessionToken index: {e}")

# CRITICAL: Index for role lookups (major bottleneck)
print("\nCreating index on _Join:users:_Role.relatedId...")
try:
    # Check if index already exists
    existing_join_related_id_index = any(idx.get('key', {}).get('relatedId') for idx in join_users_role_indexes)
    if existing_join_related_id_index:
        print("Index on _Join:users:_Role.relatedId already exists, skipping...")
    else:
        db["_Join:users:_Role"].create_index("relatedId", unique=False, background=True)
        print("✓ Index on _Join:users:_Role.relatedId created successfully")
except Exception as e:
    print(f"Error creating _Join:users:_Role.relatedId index: {e}")

# CRITICAL: Index for workspace_follower user lookups (major bottleneck)
print("\nCreating index on workspace_follower._p_user...")
try:
    # Check if index already exists
    existing_workspace_follower_user_index = any(idx.get('key', {}).get('_p_user') for idx in workspace_follower_indexes)
    if existing_workspace_follower_user_index:
        print("Index on workspace_follower._p_user already exists, skipping...")
    else:
        db["workspace_follower"].create_index("_p_user", unique=False, background=True)
        print("✓ Index on workspace_follower._p_user created successfully")
except Exception as e:
    print(f"Error creating workspace_follower._p_user index: {e}")

# CRITICAL: Index for workspace_follower workspace lookups
print("\nCreating index on workspace_follower._p_workspace...")
try:
    # Check if index already exists
    existing_workspace_follower_workspace_index = any(idx.get('key', {}).get('_p_workspace') for idx in workspace_follower_indexes)
    if existing_workspace_follower_workspace_index:
        print("Index on workspace_follower._p_workspace already exists, skipping...")
    else:
        db["workspace_follower"].create_index("_p_workspace", unique=False, background=True)
        print("✓ Index on workspace_follower._p_workspace created successfully")
except Exception as e:
    print(f"Error creating workspace_follower._p_workspace index: {e}")

# CRITICAL: Compound index for workspace_follower user + workspace (for includes)
print("\nCreating compound index on workspace_follower._p_user + _p_workspace...")
try:
    # Check if compound index already exists
    existing_compound_index = any(
        idx.get('key', {}).get('_p_user') and idx.get('key', {}).get('_p_workspace') 
        for idx in workspace_follower_indexes
    )
    if existing_compound_index:
        print("Compound index on workspace_follower._p_user + _p_workspace already exists, skipping...")
    else:
        db["workspace_follower"].create_index([("_p_user", 1), ("_p_workspace", 1)], unique=False, background=True)
        print("✓ Compound index on workspace_follower._p_user + _p_workspace created successfully")
except Exception as e:
    print(f"Error creating workspace_follower compound index: {e}")

# Index for _Role lookups by name
print("\nCreating index on _Role.name...")
try:
    # Check if index already exists
    existing_role_name_index = any(idx.get('key', {}).get('name') for idx in role_indexes)
    if existing_role_name_index:
        print("Index on _Role.name already exists, skipping...")
    else:
        db["_Role"].create_index("name", unique=False, background=True)
        print("✓ Index on _Role.name created successfully")
except Exception as e:
    print(f"Error creating _Role.name index: {e}")

print("\nIndex creation process completed!")
print("\nExpected performance improvements:")
print("- Role lookups: ~80-90% faster (was scanning _Join:users:_Role)")
print("- Workspace lookups: ~80-90% faster (was scanning workspace_follower)")
print("- Overall auth time: Should drop from 500-600ms to ~100-150ms")
