#!/usr/bin/env python3
"""
Add Organization Pointers to WorkSpaces

This script:
1. Finds all WorkSpaces
2. Gets the workspace owner (user)
3. Finds the user's organization_id
4. Adds organization pointer to the workspace (both full pointer and _p_ shorthand)

This creates the reverse link: WorkSpace → Organization

Usage:
  # Process all workspaces
  poetry run python scripts/add_organization_to_workspaces.py
  
  # Test with limited number
  poetry run python scripts/add_organization_to_workspaces.py --limit 10
  
  # Only process workspaces without organization pointer
  poetry run python scripts/add_organization_to_workspaces.py --only-new
"""

import os
import sys
from datetime import datetime, timezone
from pymongo import MongoClient
from dotenv import find_dotenv, load_dotenv
from os import environ as env
import logging
import certifi
from typing import Optional
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


class AddOrganizationToWorkspaces:
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
        self.workspaces = self.db["WorkSpace"]
        self.users = self.db["_User"]
        self.organizations = self.db["Organization"]
        
        logger.info(f"Connected to MongoDB: {self.db.name}")
        if self.limit:
            logger.info(f"Limit: {self.limit} workspaces")
        if self.only_new:
            logger.info("Only processing workspaces without organization pointer")
    
    def preflight_check(self):
        """Show stats about what will be updated"""
        logger.info("\n=== Pre-flight Check ===")
        
        # Count total workspaces
        total_workspaces = self.workspaces.count_documents({})
        
        # Count workspaces with organization pointer already
        with_org_pointer = self.workspaces.count_documents({
            "organization": {"$exists": True}
        })
        
        # Count workspaces without organization pointer
        without_org_pointer = self.workspaces.count_documents({
            "organization": {"$exists": False}
        })
        
        # Count total organizations
        total_orgs = self.organizations.count_documents({})
        
        # Count users with organization_id
        users_with_org = self.users.count_documents({
            "organization_id": {"$exists": True}
        })
        
        logger.info(f"Total WorkSpaces: {total_workspaces}")
        logger.info(f"  ✓ Already have organization pointer: {with_org_pointer}")
        logger.info(f"  ⚠️  Missing organization pointer: {without_org_pointer}")
        logger.info(f"\nTotal Organizations: {total_orgs}")
        logger.info(f"Total Users with organization_id: {users_with_org}")
        
        if self.only_new:
            logger.info(f"\n🎯 Will process: {without_org_pointer} workspaces (--only-new flag)")
        elif self.limit:
            logger.info(f"\n🎯 Will process: {min(self.limit, total_workspaces)} workspaces (--limit {self.limit})")
        else:
            logger.info(f"\n🎯 Will process: {total_workspaces} workspaces (all)")
            if with_org_pointer > 0:
                logger.warning(f"   ⚠️  WARNING: {with_org_pointer} workspaces already have organization pointer!")
                logger.warning(f"   ⚠️  Use --only-new flag to skip already updated workspaces")
        
        logger.info("=" * 60)
    
    def get_workspace_owner_user_id(self, workspace: dict) -> Optional[str]:
        """Extract owner user ID from workspace document"""
        # Try _p_user first (shorthand pointer)
        user_pointer = workspace.get("_p_user")
        if user_pointer and isinstance(user_pointer, str):
            if user_pointer.startswith("_User$"):
                return user_pointer.replace("_User$", "")
        
        # Try user object (full pointer)
        user_obj = workspace.get("user")
        if user_obj:
            if isinstance(user_obj, dict):
                return user_obj.get("objectId")
            elif isinstance(user_obj, str):
                return user_obj
        
        return None
    
    def add_organization_pointers(self):
        """Add organization pointers to workspaces"""
        logger.info("\n=== Adding Organization Pointers to WorkSpaces ===")
        
        # Build query
        query = {}
        if self.only_new:
            # Only process workspaces without organization pointer
            query["organization"] = {"$exists": False}
        
        # Find workspaces
        if self.limit:
            workspaces = list(self.workspaces.find(query).limit(self.limit))
            logger.info(f"Processing {len(workspaces)} workspaces (limited to {self.limit})")
        else:
            workspaces = list(self.workspaces.find(query))
            logger.info(f"Processing {len(workspaces)} workspaces")
        
        updated_count = 0
        skipped_no_owner = 0
        skipped_no_org = 0
        skipped_already_has = 0
        
        for workspace in workspaces:
            workspace_id = workspace["_id"]
            
            # Skip if already has organization pointer (when not using --only-new)
            if workspace.get("organization") and not self.only_new:
                logger.debug(f"⊙ WorkSpace {workspace_id} already has organization pointer")
                skipped_already_has += 1
                continue
            
            # Get workspace owner user ID
            owner_user_id = self.get_workspace_owner_user_id(workspace)
            if not owner_user_id:
                logger.warning(f"⚠️  WorkSpace {workspace_id} has no owner user")
                skipped_no_owner += 1
                continue
            
            # Get user document
            user = self.users.find_one({"_id": owner_user_id})
            if not user:
                logger.warning(f"⚠️  User {owner_user_id} not found for workspace {workspace_id}")
                skipped_no_owner += 1
                continue
            
            # Get organization_id from user
            org_id = user.get("organization_id")
            if not org_id:
                logger.debug(f"⊙ User {owner_user_id} has no organization_id (workspace {workspace_id})")
                skipped_no_org += 1
                continue
            
            # Verify organization exists
            org = self.organizations.find_one({"_id": org_id})
            if not org:
                logger.warning(f"⚠️  Organization {org_id} not found for workspace {workspace_id}")
                skipped_no_org += 1
                continue
            
            # Add organization pointer to workspace
            update_doc = {
                "$set": {
                    "organization": {
                        "__type": "Pointer",
                        "className": "Organization",
                        "objectId": org_id
                    },
                    "_p_organization": f"Organization${org_id}",  # Shorthand for Parse queries
                    "_updated_at": datetime.now(timezone.utc)
                }
            }
            
            try:
                self.workspaces.update_one(
                    {"_id": workspace_id},
                    update_doc
                )
                logger.info(f"✓ Added organization {org_id} to workspace {workspace_id}")
                updated_count += 1
            except Exception as e:
                logger.error(f"❌ Failed to update workspace {workspace_id}: {e}")
        
        logger.info(f"\n=== Summary ===")
        logger.info(f"Updated: {updated_count}")
        logger.info(f"Skipped (no owner): {skipped_no_owner}")
        logger.info(f"Skipped (no organization): {skipped_no_org}")
        if skipped_already_has > 0:
            logger.info(f"Skipped (already has org pointer): {skipped_already_has}")
    
    def verify_results(self):
        """Verify the migration results"""
        logger.info("\n=== Verification ===")
        
        # Count workspaces with organization pointer
        with_org = self.workspaces.count_documents({
            "organization": {"$exists": True}
        })
        
        # Count workspaces with _p_organization
        with_p_org = self.workspaces.count_documents({
            "_p_organization": {"$exists": True}
        })
        
        # Count workspaces without organization pointer
        without_org = self.workspaces.count_documents({
            "organization": {"$exists": False}
        })
        
        logger.info(f"WorkSpaces with organization pointer: {with_org}")
        logger.info(f"WorkSpaces with _p_organization: {with_p_org}")
        logger.info(f"WorkSpaces without organization pointer: {without_org}")
        
        if with_org != with_p_org:
            logger.warning(f"⚠️  Mismatch: {with_org} have 'organization' but {with_p_org} have '_p_organization'")
        else:
            logger.info("✓ All workspaces with organization pointer also have _p_organization")
    
    def run(self):
        """Run the migration"""
        logger.info("=" * 60)
        logger.info("Starting Add Organization to WorkSpaces")
        logger.info("=" * 60)
        
        try:
            # Step 1: Pre-flight check
            self.preflight_check()
            
            # Step 2: Add organization pointers
            self.add_organization_pointers()
            
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
        description="Add organization pointers to WorkSpace documents"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of workspaces to process (useful for testing, e.g., --limit 10)"
    )
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="Only process workspaces without organization pointer (safe for re-running)"
    )
    
    args = parser.parse_args()
    
    migration = AddOrganizationToWorkspaces(limit=args.limit, only_new=args.only_new)
    migration.run()


if __name__ == "__main__":
    main()

