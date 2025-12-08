#!/usr/bin/env python3
"""
Add Organization and Namespace Pointers to DeveloperUser

This script:
1. Finds all DeveloperUser documents
2. Gets the developer pointer (points to _User)
3. Gets the developer's organization_id and finds the organization
4. Gets the organization's default_namespace
5. Adds both organization and namespace pointers to DeveloperUser
   (includes both full pointer and _p_ shorthand for Parse Server)

Usage:
  # Process all developer users
  poetry run python scripts/add_organization_to_developer_users.py
  
  # Test with limited number
  poetry run python scripts/add_organization_to_developer_users.py --limit 10
  
  # Only process developer users without organization pointer
  poetry run python scripts/add_organization_to_developer_users.py --only-new
"""

import os
import sys
from datetime import datetime, timezone
from pymongo import MongoClient
from dotenv import find_dotenv, load_dotenv
from os import environ as env
import logging
import certifi
from typing import Optional, Dict
import argparse

# Add parent directory to path to import services
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.url_utils import clean_url

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

# MongoDB connection
MONGO_URI = clean_url(env.get("MONGO_URI") or env.get("MONGODB_URI") or env.get("MONGODB_URL"))
if not MONGO_URI:
    logger.error("MONGO_URI, MONGODB_URI, or MONGODB_URL not set in environment")
    sys.exit(1)


class AddOrganizationToDeveloperUsers:
    def __init__(self, limit: Optional[int] = None, only_new: bool = False):
        # Connect to MongoDB
        try:
            self.client = MongoClient(
                MONGO_URI, 
                serverSelectionTimeoutMS=10000,
                tlsCAFile=certifi.where()
            )
            # Test connection
            self.client.admin.command('ping')
            logger.info("✅ Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        
        self.db = self.client.get_default_database()
        self.limit = limit
        self.only_new = only_new
        
        # Collections
        self.developer_users = self.db["DeveloperUser"]
        self.users = self.db["_User"]
        self.organizations = self.db["Organization"]
        self.namespaces = self.db["Namespace"]
        
        logger.info(f"Connected to MongoDB: {self.db.name}")
        if self.limit:
            logger.info(f"Limit: {self.limit} developer users")
        if self.only_new:
            logger.info("Only processing developer users without organization pointer")
    
    def preflight_check(self):
        """Show stats about what will be updated"""
        logger.info("\n=== Pre-flight Check ===")
        
        # Count total developer users
        total_dev_users = self.developer_users.count_documents({})
        
        # Count developer users with organization pointer already
        with_org_pointer = self.developer_users.count_documents({
            "organization": {"$exists": True}
        })
        
        # Count developer users without organization pointer
        without_org_pointer = self.developer_users.count_documents({
            "organization": {"$exists": False}
        })
        
        # Count developer users with namespace pointer
        with_namespace_pointer = self.developer_users.count_documents({
            "namespace": {"$exists": True}
        })
        
        # Count total organizations
        total_orgs = self.organizations.count_documents({})
        
        # Count total namespaces
        total_namespaces = self.namespaces.count_documents({})
        
        # Count users with organization_id
        users_with_org = self.users.count_documents({
            "organization_id": {"$exists": True}
        })
        
        logger.info(f"Total DeveloperUser documents: {total_dev_users}")
        logger.info(f"  ✓ Already have organization pointer: {with_org_pointer}")
        logger.info(f"  ✓ Already have namespace pointer: {with_namespace_pointer}")
        logger.info(f"  ⚠️  Missing organization pointer: {without_org_pointer}")
        logger.info(f"\nTotal Organizations: {total_orgs}")
        logger.info(f"Total Namespaces: {total_namespaces}")
        logger.info(f"Total _User with organization_id: {users_with_org}")
        
        if self.only_new:
            logger.info(f"\n🎯 Will process: {without_org_pointer} developer users (--only-new flag)")
        elif self.limit:
            logger.info(f"\n🎯 Will process: {min(self.limit, total_dev_users)} developer users (--limit {self.limit})")
        else:
            logger.info(f"\n🎯 Will process: {total_dev_users} developer users (all)")
            if with_org_pointer > 0:
                logger.warning(f"   ⚠️  WARNING: {with_org_pointer} developer users already have organization pointer!")
                logger.warning(f"   ⚠️  Use --only-new flag to skip already updated developer users")
        
        logger.info("=" * 60)
    
    def get_developer_user_id(self, dev_user: dict) -> Optional[str]:
        """Extract developer user ID from DeveloperUser document"""
        # Try developer object (full pointer)
        developer = dev_user.get("developer")
        if developer:
            if isinstance(developer, dict):
                return developer.get("objectId")
            elif isinstance(developer, str):
                return developer
        
        # Try user object (alternative field name)
        user = dev_user.get("user")
        if user:
            if isinstance(user, dict):
                return user.get("objectId")
            elif isinstance(user, str):
                return user
        
        # Try _p_developer (shorthand pointer)
        p_developer = dev_user.get("_p_developer")
        if p_developer and isinstance(p_developer, str):
            if p_developer.startswith("_User$"):
                return p_developer.replace("_User$", "")
        
        # Try _p_user (alternative shorthand)
        p_user = dev_user.get("_p_user")
        if p_user and isinstance(p_user, str):
            if p_user.startswith("_User$"):
                return p_user.replace("_User$", "")
        
        return None
    
    def add_organization_and_namespace_pointers(self):
        """Add organization and namespace pointers to developer users"""
        logger.info("\n=== Adding Organization & Namespace Pointers to DeveloperUser ===")
        
        # Build query
        query = {}
        if self.only_new:
            # Only process developer users without organization pointer
            query["organization"] = {"$exists": False}
        
        # Find developer users
        if self.limit:
            dev_users = list(self.developer_users.find(query).limit(self.limit))
            logger.info(f"Processing {len(dev_users)} developer users (limited to {self.limit})")
        else:
            dev_users = list(self.developer_users.find(query))
            logger.info(f"Processing {len(dev_users)} developer users")
        
        updated_count = 0
        skipped_no_developer = 0
        skipped_no_org = 0
        skipped_no_namespace = 0
        skipped_already_has = 0
        
        for dev_user in dev_users:
            dev_user_id = dev_user["_id"]
            
            # Skip if already has organization pointer (when not using --only-new)
            if dev_user.get("organization") and not self.only_new:
                logger.debug(f"⊙ DeveloperUser {dev_user_id} already has organization pointer")
                skipped_already_has += 1
                continue
            
            # Get developer (the _User who owns this DeveloperUser)
            developer_user_id = self.get_developer_user_id(dev_user)
            if not developer_user_id:
                logger.warning(f"⚠️  DeveloperUser {dev_user_id} has no developer pointer")
                skipped_no_developer += 1
                continue
            
            # Get developer user document
            developer = self.users.find_one({"_id": developer_user_id})
            if not developer:
                logger.warning(f"⚠️  Developer _User {developer_user_id} not found for DeveloperUser {dev_user_id}")
                skipped_no_developer += 1
                continue
            
            # Get organization_id from developer
            org_id = developer.get("organization_id")
            if not org_id:
                logger.debug(f"⊙ Developer {developer_user_id} has no organization_id (DeveloperUser {dev_user_id})")
                skipped_no_org += 1
                continue
            
            # Verify organization exists
            org = self.organizations.find_one({"_id": org_id})
            if not org:
                logger.warning(f"⚠️  Organization {org_id} not found for DeveloperUser {dev_user_id}")
                skipped_no_org += 1
                continue
            
            # Get default namespace from organization
            namespace_id = None
            
            # Try default_namespace_id field first (simple string)
            namespace_id = org.get("default_namespace_id")
            
            # Try default_namespace pointer if no default_namespace_id
            if not namespace_id:
                default_ns = org.get("default_namespace")
                if default_ns:
                    if isinstance(default_ns, dict):
                        namespace_id = default_ns.get("objectId")
                    elif isinstance(default_ns, str):
                        namespace_id = default_ns
            
            # Try _p_default_namespace shorthand
            if not namespace_id:
                p_default_ns = org.get("_p_default_namespace")
                if p_default_ns and isinstance(p_default_ns, str):
                    if p_default_ns.startswith("Namespace$"):
                        namespace_id = p_default_ns.replace("Namespace$", "")
            
            # If still no namespace, try to find by organization_id
            if not namespace_id:
                namespace = self.namespaces.find_one({"organization_id": org_id})
                if namespace:
                    namespace_id = namespace["_id"]
                    logger.info(f"  Found namespace {namespace_id} by organization_id lookup")
            
            if not namespace_id:
                logger.warning(f"⚠️  No namespace found for organization {org_id} (DeveloperUser {dev_user_id})")
                skipped_no_namespace += 1
                continue
            
            # Verify namespace exists
            namespace = self.namespaces.find_one({"_id": namespace_id})
            if not namespace:
                logger.warning(f"⚠️  Namespace {namespace_id} not found (DeveloperUser {dev_user_id})")
                skipped_no_namespace += 1
                continue
            
            # Build update document with both organization and namespace pointers
            update_doc = {
                "$set": {
                    # Organization pointer
                    "organization": {
                        "__type": "Pointer",
                        "className": "Organization",
                        "objectId": org_id
                    },
                    "_p_organization": f"Organization${org_id}",
                    
                    # Namespace pointer
                    "namespace": {
                        "__type": "Pointer",
                        "className": "Namespace",
                        "objectId": namespace_id
                    },
                    "_p_namespace": f"Namespace${namespace_id}",
                    
                    "_updated_at": datetime.now(timezone.utc)
                }
            }
            
            try:
                self.developer_users.update_one(
                    {"_id": dev_user_id},
                    update_doc
                )
                logger.info(f"✓ Added org {org_id} & namespace {namespace_id} to DeveloperUser {dev_user_id}")
                updated_count += 1
            except Exception as e:
                logger.error(f"❌ Failed to update DeveloperUser {dev_user_id}: {e}")
        
        logger.info(f"\n=== Summary ===")
        logger.info(f"Updated: {updated_count}")
        logger.info(f"Skipped (no developer): {skipped_no_developer}")
        logger.info(f"Skipped (no organization): {skipped_no_org}")
        logger.info(f"Skipped (no namespace): {skipped_no_namespace}")
        if skipped_already_has > 0:
            logger.info(f"Skipped (already has pointers): {skipped_already_has}")
    
    def verify_results(self):
        """Verify the migration results"""
        logger.info("\n=== Verification ===")
        
        # Count developer users with organization pointer
        with_org = self.developer_users.count_documents({
            "organization": {"$exists": True}
        })
        
        # Count developer users with _p_organization
        with_p_org = self.developer_users.count_documents({
            "_p_organization": {"$exists": True}
        })
        
        # Count developer users with namespace pointer
        with_ns = self.developer_users.count_documents({
            "namespace": {"$exists": True}
        })
        
        # Count developer users with _p_namespace
        with_p_ns = self.developer_users.count_documents({
            "_p_namespace": {"$exists": True}
        })
        
        # Count developer users without organization pointer
        without_org = self.developer_users.count_documents({
            "organization": {"$exists": False}
        })
        
        logger.info(f"DeveloperUser with organization pointer: {with_org}")
        logger.info(f"DeveloperUser with _p_organization: {with_p_org}")
        logger.info(f"DeveloperUser with namespace pointer: {with_ns}")
        logger.info(f"DeveloperUser with _p_namespace: {with_p_ns}")
        logger.info(f"DeveloperUser without organization pointer: {without_org}")
        
        if with_org != with_p_org:
            logger.warning(f"⚠️  Mismatch: {with_org} have 'organization' but {with_p_org} have '_p_organization'")
        else:
            logger.info("✓ All DeveloperUsers with organization pointer also have _p_organization")
        
        if with_ns != with_p_ns:
            logger.warning(f"⚠️  Mismatch: {with_ns} have 'namespace' but {with_p_ns} have '_p_namespace'")
        else:
            logger.info("✓ All DeveloperUsers with namespace pointer also have _p_namespace")
        
        if with_org != with_ns:
            logger.warning(f"⚠️  Mismatch: {with_org} have organization but {with_ns} have namespace")
        else:
            logger.info("✓ All DeveloperUsers have both organization and namespace pointers")
    
    def run(self):
        """Run the migration"""
        logger.info("=" * 60)
        logger.info("Starting Add Organization & Namespace to DeveloperUser")
        logger.info("=" * 60)
        
        try:
            # Step 1: Pre-flight check
            self.preflight_check()
            
            # Step 2: Add organization and namespace pointers
            self.add_organization_and_namespace_pointers()
            
            # Step 3: Verify results
            self.verify_results()
            
            logger.info("\n" + "=" * 60)
            logger.info("✓ Migration completed successfully!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"\n❌ Migration failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Add organization and namespace pointers to DeveloperUser documents"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of developer users to process (useful for testing, e.g., --limit 10)"
    )
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="Only process developer users without organization pointer (safe for re-running)"
    )
    
    args = parser.parse_args()
    
    migration = AddOrganizationToDeveloperUsers(limit=args.limit, only_new=args.only_new)
    migration.run()


if __name__ == "__main__":
    main()

