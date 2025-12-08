#!/usr/bin/env python3
"""
Migration script to add multi-tenant support to Papr Memory Server

This script:
1. Creates Organization records for existing developers
2. Creates default Namespace for each organization
3. Backfills organization_id and namespace_id on Memory records
4. Updates _User records with user_type field
5. Creates necessary indexes

Run with: poetry run python scripts/migrate_to_multi_tenant.py
"""

import os
import sys
import asyncio
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv
import logging
import certifi

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables conditionally
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    load_dotenv()

MONGO_URI = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI")
if not MONGO_URI:
    logger.error("MONGO_URI or MONGODB_URI not set in environment")
    sys.exit(1)


class MultiTenantMigration:
    def __init__(self):
        # Connect to MongoDB with proper SSL certificates
        try:
            self.client = MongoClient(
                MONGO_URI, 
                serverSelectionTimeoutMS=10000,
                tlsCAFile=certifi.where()  # Use certifi's CA bundle
            )
            # Test connection
            self.client.admin.command('ping')
            logger.info("✅ Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            logger.info("\n💡 To fix SSL issues, update your MONGO_URI:")
            logger.info("   Change: tlsInsecure=false")
            logger.info("   To:     tlsInsecure=true")
            logger.info("\n   Or remove ssl=true&tlsInsecure=false entirely")
            raise
        
        self.db = self.client.get_default_database()
        
        # Collections
        self.users = self.db["_User"]
        self.memories = self.db["Memory"]
        self.developer_users = self.db["DeveloperUser"]
        
        # New collections
        self.organizations = self.db["Organization"]
        self.namespaces = self.db["Namespace"]
        self.api_keys = self.db["APIKey"]
        
        logger.info(f"Connected to MongoDB: {self.db.name}")
    
    def create_collections(self):
        """Create new collections with validation"""
        logger.info("Creating Organization and Namespace collections...")
        
        # Organization schema
        if "Organization" not in self.db.list_collection_names():
            self.db.create_collection("Organization")
            logger.info("✓ Created Organization collection")
        
        # Namespace schema
        if "Namespace" not in self.db.list_collection_names():
            self.db.create_collection("Namespace")
            logger.info("✓ Created Namespace collection")
        
        # APIKey schema
        if "APIKey" not in self.db.list_collection_names():
            self.db.create_collection("APIKey")
            logger.info("✓ Created APIKey collection")
    
    def create_indexes(self):
        """Create indexes for performance"""
        logger.info("Creating indexes...")
        
        # Organization indexes
        self.organizations.create_index([("owner_user_id", ASCENDING)], unique=True)
        logger.info("✓ Created Organization.owner_user_id index")
        
        # Namespace indexes
        self.namespaces.create_index([("organization_id", ASCENDING)])
        self.namespaces.create_index([
            ("organization_id", ASCENDING),
            ("name", ASCENDING)
        ], unique=True)
        logger.info("✓ Created Namespace indexes")
        
        # Memory indexes (critical for multi-tenant queries)
        self.memories.create_index([
            ("organization_id", ASCENDING),
            ("namespace_id", ASCENDING),
            ("_created_at", DESCENDING)
        ])
        self.memories.create_index([
            ("organization_id", ASCENDING),
            ("_created_at", DESCENDING)
        ])
        self.memories.create_index([
            ("namespace_id", ASCENDING),
            ("_created_at", DESCENDING)
        ])
        logger.info("✓ Created Memory multi-tenant indexes")
        
        # User indexes
        self.users.create_index([("user_type", ASCENDING)])
        self.users.create_index([("organization_id", ASCENDING)])
        logger.info("✓ Created User indexes")
        
        # DeveloperUser indexes
        self.developer_users.create_index([
            ("organization_id", ASCENDING),
            ("namespace_id", ASCENDING)
        ])
        logger.info("✓ Created DeveloperUser indexes")
        
        # APIKey indexes
        self.api_keys.create_index([("key", ASCENDING)], unique=True)
        self.api_keys.create_index([("organization_id", ASCENDING)])
        self.api_keys.create_index([("namespace_id", ASCENDING)])
        logger.info("✓ Created APIKey indexes")
    
    def migrate_developers_to_organizations(self):
        """Create Organization for each developer"""
        logger.info("\n=== Migrating Developers to Organizations ===")
        
        # Find all developers (users with userAPIkey field)
        developers = self.users.find({
            "userAPIkey": {"$exists": True},
            "isDeveloper": True
        })
        
        created_count = 0
        skipped_count = 0
        
        for dev in developers:
            user_id = dev["_id"]
            
            # Check if org already exists
            existing_org = self.organizations.find_one({"owner_user_id": user_id})
            if existing_org:
                logger.info(f"⊙ Organization already exists for user {user_id[:10]}...")
                skipped_count += 1
                continue
            
            # Create organization
            org_id = f"org_{user_id[:16]}"
            org = {
                "_id": org_id,
                "name": dev.get("email", "").split("@")[0] or f"Developer {user_id[:8]}",
                "owner_user_id": user_id,
                "team_members": [user_id],
                "_created_at": dev.get("_created_at", datetime.now()),
                "_updated_at": datetime.now(),
                
                # Settings
                "settings": {
                    "default_namespace": f"ns_{org_id}_production"
                },
                
                # Plan (default to trial)
                "plan_tier": "trial",
                "subscription_id": None
            }
            
            self.organizations.insert_one(org)
            
            # Update user record
            self.users.update_one(
                {"_id": user_id},
                {
                    "$set": {
                        "user_type": "DEVELOPER",
                        "organization_id": org_id,
                        "_updated_at": datetime.now()
                    }
                }
            )
            
            logger.info(f"✓ Created Organization {org_id} for user {user_id[:10]}...")
            created_count += 1
        
        logger.info(f"\nOrganizations: {created_count} created, {skipped_count} skipped")
    
    def create_default_namespaces(self):
        """Create default namespace for each organization"""
        logger.info("\n=== Creating Default Namespaces ===")
        
        orgs = self.organizations.find()
        created_count = 0
        skipped_count = 0
        
        for org in orgs:
            org_id = org["_id"]
            ns_id = f"ns_{org_id}_production"
            
            # Check if namespace exists
            existing_ns = self.namespaces.find_one({"_id": ns_id})
            if existing_ns:
                logger.info(f"⊙ Namespace already exists: {ns_id}")
                skipped_count += 1
                continue
            
            # Create production namespace
            namespace = {
                "_id": ns_id,
                "name": f"{org['name']}-production",
                "organization_id": org_id,
                "environment_type": "production",
                "is_active": True,
                "_created_at": org.get("_created_at", datetime.now()),
                "_updated_at": datetime.now(),
                
                # Rate limits (inherit from org)
                "rate_limits": {
                    "memories_per_month": None,  # Unlimited for now
                    "api_calls_per_day": None
                }
            }
            
            self.namespaces.insert_one(namespace)
            logger.info(f"✓ Created Namespace {ns_id}")
            created_count += 1
        
        logger.info(f"\nNamespaces: {created_count} created, {skipped_count} skipped")
    
    def migrate_api_keys(self):
        """Migrate existing API keys to new schema"""
        logger.info("\n=== Migrating API Keys ===")
        
        # Find developers with API keys
        developers = self.users.find({
            "userAPIkey": {"$exists": True},
            "organization_id": {"$exists": True}
        })
        
        created_count = 0
        skipped_count = 0
        
        for dev in developers:
            api_key = dev.get("userAPIkey")
            if not api_key:
                continue
            
            # Check if already migrated
            existing = self.api_keys.find_one({"key": api_key})
            if existing:
                skipped_count += 1
                continue
            
            org_id = dev.get("organization_id")
            namespace = self.namespaces.find_one({"organization_id": org_id})
            
            if not namespace:
                logger.warning(f"No namespace found for org {org_id}")
                continue
            
            # Create APIKey record
            api_key_doc = {
                "_id": f"ak_{api_key[:16]}",
                "key": api_key,
                "name": "Production API Key (Migrated)",
                "namespace_id": namespace["_id"],
                "organization_id": org_id,
                "environment": "production",
                "permissions": ["read", "write", "delete"],
                "is_active": True,
                "_created_at": dev.get("_created_at", datetime.now()),
                "_updated_at": datetime.now(),
                "last_used_at": None
            }
            
            self.api_keys.insert_one(api_key_doc)
            logger.info(f"✓ Migrated API key for org {org_id}")
            created_count += 1
        
        logger.info(f"\nAPI Keys: {created_count} created, {skipped_count} skipped")
    
    def backfill_memory_tenant_fields(self, batch_size=1000, max_batches=None):
        """Backfill organization_id and namespace_id on Memory records"""
        logger.info("\n=== Backfilling Memory Tenant Fields ===")
        
        # Count memories without tenant fields
        total_memories = self.memories.count_documents({
            "$or": [
                {"organization_id": {"$exists": False}},
                {"namespace_id": {"$exists": False}}
            ]
        })
        
        logger.info(f"Found {total_memories} memories to migrate")
        
        if total_memories == 0:
            logger.info("✓ All memories already have tenant fields")
            return
        
        updated_count = 0
        skipped_count = 0
        batch_num = 0
        
        while True:
            if max_batches and batch_num >= max_batches:
                logger.info(f"Reached max batch limit: {max_batches}")
                break
            
            # Get batch of memories without tenant fields
            memories = list(self.memories.find(
                {
                    "$or": [
                        {"organization_id": {"$exists": False}},
                        {"namespace_id": {"$exists": False}}
                    ]
                },
                {"_id": 1, "_p_user": 1, "user": 1}
            ).limit(batch_size))
            
            if not memories:
                break
            
            batch_num += 1
            logger.info(f"\nProcessing batch {batch_num} ({len(memories)} memories)...")
            
            for memory in memories:
                memory_id = memory["_id"]
                
                # Find user_id from pointer
                user_id = None
                if "_p_user" in memory:
                    user_id = memory["_p_user"].replace("_User$", "")
                elif "user" in memory and isinstance(memory["user"], dict):
                    user_id = memory["user"].get("objectId")
                
                if not user_id:
                    logger.warning(f"⚠ Memory {memory_id} has no user, skipping")
                    skipped_count += 1
                    continue
                
                # Find user to get organization
                user = self.users.find_one({"_id": user_id})
                if not user:
                    logger.warning(f"⚠ User {user_id} not found for memory {memory_id}")
                    skipped_count += 1
                    continue
                
                # Determine organization_id
                org_id = None
                
                # If user is developer
                if user.get("organization_id"):
                    org_id = user["organization_id"]
                
                # If user is end user (has developer_organization_id)
                elif user.get("developer_organization_id"):
                    org_id = user["developer_organization_id"]
                
                # If user is end user via DeveloperUser
                elif user.get("type") == "developerUser" or user.get("user_type") == "END_USER":
                    dev_user = self.developer_users.find_one({"_p_user": f"_User${user_id}"})
                    if dev_user and "_p_developer" in dev_user:
                        developer_id = dev_user["_p_developer"].replace("_User$", "")
                        developer = self.users.find_one({"_id": developer_id})
                        if developer:
                            org_id = developer.get("organization_id")
                
                if not org_id:
                    logger.warning(f"⚠ Could not determine org for memory {memory_id}, user {user_id}")
                    skipped_count += 1
                    continue
                
                # Get default namespace for organization
                namespace = self.namespaces.find_one({"organization_id": org_id})
                if not namespace:
                    logger.warning(f"⚠ No namespace found for org {org_id}")
                    skipped_count += 1
                    continue
                
                # Update memory
                self.memories.update_one(
                    {"_id": memory_id},
                    {
                        "$set": {
                            "organization_id": org_id,
                            "namespace_id": namespace["_id"],
                            "_updated_at": datetime.now()
                        }
                    }
                )
                
                updated_count += 1
                
                if updated_count % 100 == 0:
                    logger.info(f"  Updated {updated_count} memories...")
        
        logger.info(f"\nMemories: {updated_count} updated, {skipped_count} skipped")
    
    def update_developer_users(self):
        """Add organization_id and namespace_id to DeveloperUser records"""
        logger.info("\n=== Updating DeveloperUser Records ===")
        
        dev_users = self.developer_users.find({
            "$or": [
                {"organization_id": {"$exists": False}},
                {"namespace_id": {"$exists": False}}
            ]
        })
        
        updated_count = 0
        skipped_count = 0
        
        for dev_user in dev_users:
            # Get developer user_id
            developer_id = None
            if "_p_developer" in dev_user:
                developer_id = dev_user["_p_developer"].replace("_User$", "")
            
            if not developer_id:
                skipped_count += 1
                continue
            
            # Find developer's organization
            developer = self.users.find_one({"_id": developer_id})
            if not developer or "organization_id" not in developer:
                skipped_count += 1
                continue
            
            org_id = developer["organization_id"]
            
            # Get default namespace
            namespace = self.namespaces.find_one({"organization_id": org_id})
            if not namespace:
                skipped_count += 1
                continue
            
            # Update DeveloperUser
            self.developer_users.update_one(
                {"_id": dev_user["_id"]},
                {
                    "$set": {
                        "organization_id": org_id,
                        "namespace_id": namespace["_id"],
                        "_updated_at": datetime.now()
                    }
                }
            )
            
            # Also update the _User record for this end user
            if "_p_user" in dev_user:
                end_user_id = dev_user["_p_user"].replace("_User$", "")
                self.users.update_one(
                    {"_id": end_user_id},
                    {
                        "$set": {
                            "user_type": "END_USER",
                            "developer_organization_id": org_id,
                            "_updated_at": datetime.now()
                        }
                    }
                )
            
            updated_count += 1
        
        logger.info(f"DeveloperUsers: {updated_count} updated, {skipped_count} skipped")
    
    def verify_migration(self):
        """Verify migration completed successfully"""
        logger.info("\n=== Verifying Migration ===")
        
        stats = {
            "organizations": self.organizations.count_documents({}),
            "namespaces": self.namespaces.count_documents({}),
            "api_keys": self.api_keys.count_documents({}),
            "memories_with_tenant": self.memories.count_documents({
                "organization_id": {"$exists": True},
                "namespace_id": {"$exists": True}
            }),
            "memories_without_tenant": self.memories.count_documents({
                "$or": [
                    {"organization_id": {"$exists": False}},
                    {"namespace_id": {"$exists": False}}
                ]
            }),
            "developers": self.users.count_documents({"user_type": "DEVELOPER"}),
            "end_users": self.users.count_documents({"user_type": "END_USER"}),
        }
        
        logger.info("\nMigration Statistics:")
        logger.info(f"  Organizations: {stats['organizations']}")
        logger.info(f"  Namespaces: {stats['namespaces']}")
        logger.info(f"  API Keys: {stats['api_keys']}")
        logger.info(f"  Memories (migrated): {stats['memories_with_tenant']}")
        logger.info(f"  Memories (not migrated): {stats['memories_without_tenant']}")
        logger.info(f"  Developers: {stats['developers']}")
        logger.info(f"  End Users: {stats['end_users']}")
        
        # Warnings
        if stats['memories_without_tenant'] > 0:
            logger.warning(f"\n⚠ WARNING: {stats['memories_without_tenant']} memories still lack tenant fields")
            logger.warning("  Run migration again or use max_batches=None to process all")
        
        if stats['organizations'] == 0:
            logger.error("\n❌ ERROR: No organizations created!")
        
        logger.info("\n✓ Migration verification complete")
    
    def run_migration(self, backfill_batch_size=1000, backfill_max_batches=None):
        """Run complete migration"""
        logger.info("=" * 60)
        logger.info("Starting Multi-Tenant Migration")
        logger.info("=" * 60)
        
        try:
            # Step 1: Create collections
            self.create_collections()
            
            # Step 2: Create indexes
            self.create_indexes()
            
            # Step 3: Migrate developers to organizations
            self.migrate_developers_to_organizations()
            
            # Step 4: Create default namespaces
            self.create_default_namespaces()
            
            # Step 5: Migrate API keys
            self.migrate_api_keys()
            
            # Step 6: Update DeveloperUser records
            self.update_developer_users()
            
            # Step 7: Backfill Memory tenant fields (can be run in batches)
            self.backfill_memory_tenant_fields(
                batch_size=backfill_batch_size,
                max_batches=backfill_max_batches
            )
            
            # Step 8: Verify
            self.verify_migration()
            
            logger.info("\n" + "=" * 60)
            logger.info("✓ Migration completed successfully!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"\n❌ Migration failed: {e}")
            raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate Papr to multi-tenant architecture")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of memories to process per batch (default: 1000)"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to process (default: None = all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually make changes, just show what would happen"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
        # TODO: Implement dry-run logic
        return
    
    migration = MultiTenantMigration()
    migration.run_migration(
        backfill_batch_size=args.batch_size,
        backfill_max_batches=args.max_batches
    )


if __name__ == "__main__":
    main()

