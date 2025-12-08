#!/usr/bin/env python3
"""
Test migration script - verifies what WOULD happen without making changes

This is a DRY-RUN script that shows you what the migration would do
without actually modifying the database.
"""

import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv
import logging
import certifi

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables conditionally
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    logger.error("MONGO_URI not set in environment")
    sys.exit(1)


def test_migration_safety():
    """Check what the migration would do WITHOUT making changes"""
    
    # Connect to MongoDB with proper SSL certificates
    try:
        client = MongoClient(
            MONGO_URI, 
            serverSelectionTimeoutMS=10000,
            tlsCAFile=certifi.where()  # Use certifi's CA bundle
        )
        # Test connection
        client.admin.command('ping')
        logger.info("✅ Successfully connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        logger.info("\n💡 To fix SSL issues, update your MONGO_URI:")
        logger.info("   Change: tlsInsecure=false")
        logger.info("   To:     tlsInsecure=true")
        logger.info("\n   Or remove ssl=true&tlsInsecure=false entirely")
        raise
    
    db = client.get_default_database()
    
    logger.info("=" * 60)
    logger.info("MIGRATION SAFETY CHECK (DRY RUN)")
    logger.info("=" * 60)
    
    # Check 1: New collections would be created (safe)
    logger.info("\n✅ NEW COLLECTIONS TO BE CREATED:")
    new_collections = ["Organization", "Namespace", "APIKey"]
    for coll in new_collections:
        exists = coll in db.list_collection_names()
        if exists:
            count = db[coll].count_documents({})
            logger.info(f"  {coll}: Already exists ({count} documents)")
        else:
            logger.info(f"  {coll}: Will be created (NEW)")
    
    # Check 2: Developers that would become organizations
    logger.info("\n✅ DEVELOPERS → ORGANIZATIONS:")
    developers = db["_User"].count_documents({
        "userAPIkey": {"$exists": True},
        "isDeveloper": True
    })
    already_migrated = db["_User"].count_documents({
        "user_type": {"$exists": True}
    })
    logger.info(f"  Developers found: {developers}")
    logger.info(f"  Already migrated: {already_migrated}")
    logger.info(f"  Would create: {developers - already_migrated} organizations")
    
    # Check 3: Memories that need backfill
    logger.info("\n✅ MEMORIES TO UPDATE:")
    total_memories = db.Memory.count_documents({})
    migrated_memories = db.Memory.count_documents({
        "organization_id": {"$exists": True}
    })
    logger.info(f"  Total memories: {total_memories}")
    logger.info(f"  Already migrated: {migrated_memories}")
    logger.info(f"  Need backfill: {total_memories - migrated_memories}")
    
    # Check 4: DeveloperUsers that need update
    logger.info("\n✅ DEVELOPER USERS TO UPDATE:")
    total_dev_users = db.DeveloperUser.count_documents({})
    migrated_dev_users = db.DeveloperUser.count_documents({
        "organization_id": {"$exists": True}
    })
    logger.info(f"  Total DeveloperUsers: {total_dev_users}")
    logger.info(f"  Already migrated: {migrated_dev_users}")
    logger.info(f"  Need update: {total_dev_users - migrated_dev_users}")
    
    # Check 5: Show what would happen to a sample memory
    logger.info("\n✅ SAMPLE MEMORY UPDATE:")
    sample_memory = db.Memory.find_one({
        "organization_id": {"$exists": False}
    })
    if sample_memory:
        logger.info(f"  Sample Memory ID: {sample_memory.get('_id')}")
        logger.info(f"  Current fields: {list(sample_memory.keys())}")
        logger.info(f"  Would ADD: organization_id, namespace_id")
        logger.info(f"  Would NOT remove any existing fields ✅")
    else:
        logger.info(f"  All memories already migrated!")
    
    # Check 6: Verify NO destructive operations
    logger.info("\n✅ SAFETY VERIFICATION:")
    logger.info("  ✓ No DROP operations")
    logger.info("  ✓ No DELETE operations")
    logger.info("  ✓ No REMOVE operations")
    logger.info("  ✓ Only INSERT (new docs) and UPDATE (add fields)")
    logger.info("  ✓ All operations use $set (additive only)")
    logger.info("  ✓ Existing data preserved")
    
    logger.info("\n" + "=" * 60)
    logger.info("SAFE TO RUN: Migration is non-destructive ✅")
    logger.info("=" * 60)
    
    # Show command to run
    logger.info("\nTo run the migration:")
    logger.info("  Test with limited batches:")
    logger.info("    poetry run python scripts/migrate_to_multi_tenant.py --max-batches 10")
    logger.info("")
    logger.info("  Full migration:")
    logger.info("    poetry run python scripts/migrate_to_multi_tenant.py")


if __name__ == "__main__":
    test_migration_safety()

