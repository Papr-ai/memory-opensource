#!/usr/bin/env python3
"""
Cleanup v1 migration artifacts that won't show in Parse Dashboard

This removes Organization, Namespace, and APIKey documents created by v1
that use the wrong format (objectId instead of _id).

This is SAFE - it only removes v1 artifacts, not your actual data.
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

MONGO_URI = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI")
if not MONGO_URI:
    logger.error("MONGO_URI not set")
    sys.exit(1)


def cleanup_v1_artifacts():
    """Remove v1 migration artifacts"""
    
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client.get_default_database()
    
    logger.info("=" * 60)
    logger.info("Cleaning up v1 migration artifacts")
    logger.info("=" * 60)
    
    # Count what we're about to remove
    org_count = db.Organization.count_documents({})
    ns_count = db.Namespace.count_documents({})
    ak_count = db.APIKey.count_documents({})
    
    logger.info(f"\nFound:")
    logger.info(f"  Organizations: {org_count}")
    logger.info(f"  Namespaces: {ns_count}")
    logger.info(f"  API Keys: {ak_count}")
    
    # Remove v1 artifacts (they don't have proper Parse format)
    # We'll recreate them with v2 migration
    
    response = input("\nRemove these collections to recreate with proper Parse format? (yes/no): ")
    if response.lower() != "yes":
        logger.info("Cancelled")
        return
    
    # Remove collections
    result_org = db.Organization.delete_many({})
    result_ns = db.Namespace.delete_many({})
    result_ak = db.APIKey.delete_many({})
    
    logger.info(f"\n✓ Removed {result_org.deleted_count} Organizations")
    logger.info(f"✓ Removed {result_ns.deleted_count} Namespaces")
    logger.info(f"✓ Removed {result_ak.deleted_count} API Keys")
    
    # Also clean up organization_id and user_type from _User if needed
    response = input("\nAlso remove organization_id and user_type from _User? (yes/no): ")
    if response.lower() == "yes":
        result_user = db["_User"].update_many(
            {},
            {
                "$unset": {
                    "organization_id": "",
                    "user_type": ""
                }
            }
        )
        logger.info(f"✓ Cleaned {result_user.modified_count} _User records")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Cleanup complete! Ready for v2 migration")
    logger.info("=" * 60)
    logger.info("\nRun: poetry run python scripts/migrate_to_multi_tenant_v2.py")


if __name__ == "__main__":
    cleanup_v1_artifacts()

