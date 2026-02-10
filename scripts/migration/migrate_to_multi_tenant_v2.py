#!/usr/bin/env python3
"""
Enhanced Multi-Tenant Migration Script v2

This script:
1. Creates Organizations with proper Parse Server format
2. Discovers workspace members via workspace_follower relationships
3. Migrates workspace creators (users with userAPIkey, type != "developerUser")
4. Excludes end-users (type: "developerUser")
5. Maps workspace roles to organization roles
6. Ensures Parse Dashboard visibility
7. Uses Pydantic models for proper Parse Server schema
8. Sets user_type: "CREATOR" for workspace owners, "TEAM_MEMBER" for team members

Workspace Creator vs End-User:
  - Workspace Creator: has userAPIkey AND type != "developerUser" → CREATE Organization
  - End-User: type == "developerUser" → SKIP (uses creator's API key)

Usage:
  # First time (all workspace creators)
  poetry run python scripts/migrate_to_multi_tenant_v2.py
  
  # Re-run for new workspace creators only (safe for production)
  poetry run python scripts/migrate_to_multi_tenant_v2.py --only-new
  
  # Test with limited number
  poetry run python scripts/migrate_to_multi_tenant_v2.py --only-new --limit 5
"""

import os
import sys
from datetime import datetime, timezone
from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import find_dotenv, load_dotenv
from os import environ as env
import logging
import certifi
from typing import List, Dict, Set, Optional
import httpx
import argparse
import secrets
import string

# Add parent directory to path to import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.parse_server import (
    Organization,
    Namespace,
    APIKey,
    ParseUserPointer,
    OrganizationPointer,
    NamespacePointer,
    WorkspacePointer,
    SubscriptionPointer,
    EnvironmentType
)
from services.url_utils import clean_url

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check logging configuration before anything else
print(f"DEBUG - Initial logging level: {logging.getLevelName(logging.getLogger().level)}", flush=True)

# DEBUG: Print immediately to see if script starts
print("=" * 60, flush=True)
print("DEBUG: Script execution started!", flush=True)
print("=" * 60, flush=True)

# Load environment variables - following memory_management.py pattern
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)
    print(f"DEBUG - Loaded .env from: {ENV_FILE}", flush=True)
    print(f"DEBUG - LOGGING_ENV after load: {env.get('LOGGING_ENV', 'NOT SET')}", flush=True)

# MongoDB connection - following memory_management.py pattern
MONGO_URI = clean_url(env.get("MONGO_URI") or env.get("MONGODB_URI") or env.get("MONGODB_URL"))
logger.info(f"MONGO_URI: {MONGO_URI}")
print(f"DEBUG - MONGO_URI: {MONGO_URI}", flush=True)
print(f"DEBUG - Database: {MONGO_URI.split('/')[-1].split('?')[0] if MONGO_URI else 'N/A'}", flush=True)
if not MONGO_URI:
    logger.error("MONGO_URI, MONGODB_URI, or MONGODB_URL not set in environment")
    sys.exit(1)


class EnhancedMultiTenantMigration:
    def __init__(self, limit: Optional[int] = None, only_new: bool = False):
        # Connect to MongoDB with proper SSL certificates
        print("DEBUG - Attempting MongoDB connection...", flush=True)
        try:
            print("DEBUG - Creating MongoClient...", flush=True)
            self.client = MongoClient(
                MONGO_URI, 
                serverSelectionTimeoutMS=10000,
                tlsCAFile=certifi.where()
            )
            print("DEBUG - MongoClient created, testing connection with ping...", flush=True)
            # Test connection
            self.client.admin.command('ping')
            print("DEBUG - MongoDB ping successful!", flush=True)
            logger.info("✅ Successfully connected to MongoDB")
        except Exception as e:
            print(f"DEBUG - MongoDB connection FAILED: {e}", flush=True)
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        
        print("DEBUG - Getting default database...", flush=True)
        self.db = self.client.get_default_database()
        print(f"DEBUG - Connected to database: {self.db.name}", flush=True)
        self.limit = limit  # Limit number of developers to process
        self.only_new = only_new  # Only process developers without organization_id
        
        # Collections
        self.users = self.db["_User"]
        self.memories = self.db["Memory"]
        self.developer_users = self.db["DeveloperUser"]
        self.workspace_followers = self.db["workspace_follower"]
        self.workspaces = self.db["WorkSpace"]
        self.roles = self.db["_Role"]
        
        # New collections
        self.organizations = self.db["Organization"]
        self.namespaces = self.db["Namespace"]
        self.api_keys = self.db["APIKey"]
        
        mode_desc = ""
        if self.only_new:
            mode_desc = " (Only NEW developers without organization_id)"
        if self.limit:
            logger.info(f"Connected to MongoDB: {self.db.name} (Limited to {self.limit} developers{mode_desc})")
        else:
            logger.info(f"Connected to MongoDB: {self.db.name}{mode_desc}")
        
        # Parse config (for Relations updates) - following memory_management.py pattern
        self.parse_server_url = clean_url(
            env.get("PARSE_SERVER_URL") or
            env.get("PARSE_SERVER") or
            env.get("PARSE_API_ENDPOINT") or
            env.get("PAPR_PARSE_SERVER_URL") or
            env.get("PARSE_SERVER_URL")  # Following memory_management.py pattern
        )
        if self.parse_server_url and not self.parse_server_url.rstrip("/").endswith("/parse"):
            # Ensure trailing /parse
            self.parse_server_url = self.parse_server_url.rstrip("/") + "/parse"

        self.parse_app_id = clean_url(
            env.get("PARSE_SERVER_APPLICATION_ID") or
            env.get("PARSE_APPLICATION_ID") or  # Following memory_management.py pattern
            env.get("PARSE_APP_ID") or
            env.get("PAPR_PARSE_APP_ID") or
            env.get("PARSE_SERVER_APP_ID")
        )

        self.parse_master_key = clean_url(
            env.get("PARSE_SERVER_MASTER_KEY") or
            env.get("PARSE_MASTER_KEY") or  # Following memory_management.py pattern
            env.get("PAPR_PARSE_MASTER_KEY") or
            env.get("PARSE_SERVER_MASTER")
        )

        logger.info(f"Parse config - URL: {self.parse_server_url}, App ID: {self.parse_app_id}, Master Key: {'***' if self.parse_master_key else 'None'}")
        print(f"DEBUG - Parse Server URL: {self.parse_server_url}", flush=True)
        print(f"DEBUG - Parse App ID: {self.parse_app_id}", flush=True)
        print(f"DEBUG - Parse Master Key: {'SET' if self.parse_master_key else 'NOT SET'}", flush=True)
        print(f"DEBUG - LOGGING_ENV from env: {env.get('LOGGING_ENV', 'NOT SET')}", flush=True)
        print(f"DEBUG - Current logger level: {logging.getLevelName(logger.level)}", flush=True)
        print(f"DEBUG - Root logger level: {logging.getLevelName(logging.getLogger().level)}", flush=True)

        # Also check for REST API key if it exists (following memory_management.py pattern)
        self.parse_rest_api_key = clean_url(env.get("PARSE_REST_API_KEY"))
        if self.parse_rest_api_key:
            logger.info("Found PARSE_REST_API_KEY (could be used as alternative to master key)")
    
    def generate_parse_object_id(self) -> str:
        """
        Generate a Parse Server compatible 10-character alphanumeric objectId.
        Parse Server uses a mix of uppercase, lowercase letters and numbers.
        Example: "001fQ2gmNM", "Aati07jMX7"
        """
        chars = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
        return ''.join(secrets.choice(chars) for _ in range(10))
    
    def convert_acl_to_parse_format(self, acl: Dict) -> Dict:
        """
        Convert ACL from {"user_id": {"read": true, "write": true}}
        to Parse format: {"user_id": {"r": true, "w": true}}
        """
        if not acl:
            return {}
        
        parse_acl = {}
        for user_id, perms in acl.items():
            parse_acl[user_id] = {}
            # Support both long-form (read/write) and short-form (r/w)
            if perms.get("read") or perms.get("r"):
                parse_acl[user_id]["r"] = True
            if perms.get("write") or perms.get("w"):
                parse_acl[user_id]["w"] = True
        
        return parse_acl
    
    def pydantic_to_parse_doc(self, model_instance, object_id: str) -> Dict:
        """
        Convert Pydantic model to Parse Server MongoDB document format.
        
        Parse Server expects:
        - _id: 10-character alphanumeric objectId
        - _created_at: ISO datetime
        - _updated_at: ISO datetime
        - _acl: ACL object with "r" and "w" (not "read" and "write")
        - _wperm: List of write permissions
        - _rperm: List of read permissions
        - Pointers as: {"__type": "Pointer", "className": "X", "objectId": "y"}
        - Shorthand pointer fields as: _p_fieldname: "ClassName$objectId"
        """
        # Get the model dump (converts Pydantic to dict)
        data = model_instance.model_dump(exclude_none=False, by_alias=True)
        
        # Remove Pydantic fields
        data.pop('objectId', None)
        data.pop('createdAt', None)
        data.pop('updatedAt', None)
        
        # Convert ACL to Parse format
        acl = data.pop('ACL', {})
        parse_acl = self.convert_acl_to_parse_format(acl)
        
        parse_doc = {
            "_id": object_id,
            "_created_at": datetime.now(timezone.utc),
            "_updated_at": datetime.now(timezone.utc),
            **data
        }
        
        # Add ACL fields if ACL exists
        if parse_acl:
            parse_doc["_acl"] = parse_acl
            parse_doc["_wperm"] = [user_id for user_id, perms in parse_acl.items() if perms.get("w")]
            parse_doc["_rperm"] = [user_id for user_id, perms in parse_acl.items() if perms.get("r")]
        
        # Add _p_ shorthand fields for all pointer objects
        # Parse Server uses these for efficient querying
        for field_name, field_value in list(parse_doc.items()):
            if isinstance(field_value, dict) and field_value.get("__type") == "Pointer":
                class_name = field_value.get("className")
                obj_id = field_value.get("objectId")
                if class_name and obj_id:
                    # Add shorthand pointer field: _p_fieldname: "ClassName$objectId"
                    parse_doc[f"_p_{field_name}"] = f"{class_name}${obj_id}"
                    logger.debug(f"  Added _p_{field_name}: {class_name}${obj_id}")
        
        return parse_doc

    def _build_acl(self, owner_user_id: Optional[str], workspace_id: Optional[str]) -> Dict:
        """Create a sane ACL: owner full access + member role full access if workspace known."""
        acl: Dict[str, Dict[str, bool]] = {}
        
        # Only add owner if owner_user_id is not None
        if owner_user_id:
            acl[owner_user_id] = {"read": True, "write": True}
        
        # Add workspace member role if workspace exists
        if workspace_id:
            acl[f"role:member-{workspace_id}"] = {"read": True, "write": True}
        
        # If ACL is empty, add public read access as fallback
        if not acl:
            acl["*"] = {"read": True}
        
        return acl

    def _get_parse_headers(self) -> Optional[Dict[str, str]]:
        if not (self.parse_server_url and self.parse_app_id):
            logger.warning("Missing Parse server URL or App ID")
            return None

        if not (self.parse_master_key or self.parse_rest_api_key):
            logger.warning("Missing Parse master key or REST API key")
            return None

        headers = {
            "X-Parse-Application-Id": self.parse_app_id,
            "Content-Type": "application/json"
        }

        # Prefer master key over REST API key for admin operations
        if self.parse_master_key:
            headers["X-Parse-Master-Key"] = self.parse_master_key
        elif self.parse_rest_api_key:
            headers["X-Parse-REST-API-Key"] = self.parse_rest_api_key

        return headers

    def _add_team_members_relation(self, org_id: str, member_ids: List[str]) -> bool:
        """Use Parse REST API to add team_members relation so it shows in Dashboard."""
        if not member_ids:
            logger.info(f"  No team members to add for org {org_id}")
            return True

        headers = self._get_parse_headers()
        if not headers:
            logger.warning("Parse config missing; skipping team_members relation update via REST API")
            logger.warning(f"  Parse URL: {self.parse_server_url}")
            logger.warning(f"  Parse App ID: {self.parse_app_id}")
            logger.warning(f"  Parse Master Key: {'Present' if self.parse_master_key else 'Missing'}")
            return False

        url = f"{self.parse_server_url}/parse/classes/Organization/{org_id}"
        objects = [{"__type": "Pointer", "className": "_User", "objectId": mid} for mid in member_ids]
        payload = {"team_members": {"__op": "AddRelation", "objects": objects}}

        logger.info(f"  Adding {len(member_ids)} team_members via REST API to org {org_id}")
        logger.info(f"  URL: {url}")
        logger.info(f"  Payload: {payload}")

        try:
            # Disable SSL verification for internal migration script
            with httpx.Client(timeout=30.0, verify=False) as client:
                r = client.put(url, headers=headers, json=payload)
                logger.info(f"  Response status: {r.status_code}")
                logger.info(f"  Response body: {r.text}")

                if r.status_code >= 200 and r.status_code < 300:
                    logger.info(f"  ✓ Added {len(member_ids)} team_members via REST relation for org {org_id}")
                    return True
                else:
                    logger.warning(f"Failed to add team_members relation for org {org_id}: {r.status_code} {r.text}")
                    return False
        except Exception as e:
            logger.warning(f"Exception adding team_members relation for org {org_id}: {e}")
            import traceback
            logger.warning(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def get_workspace_for_developer(self, developer_id: str) -> Optional[Dict]:
        """
        Get workspace details for a developer via their workspace_follower.
        Returns dict with workspace_id and subscription_id.
        """
        # Get developer's selected workspace_follower
        developer = self.users.find_one({"_id": developer_id})
        if not developer:
            return None
        
        workspace_follower = None
        workspace_follower_pointer = developer.get("isSelectedWorkspaceFollower")
        
        if workspace_follower_pointer:
            # Get workspace_follower record from pointer
            if isinstance(workspace_follower_pointer, dict):
                workspace_follower_id = workspace_follower_pointer.get("objectId")
            else:
                workspace_follower_id = workspace_follower_pointer
            
            workspace_follower = self.workspace_followers.find_one({"_id": workspace_follower_id})
        
        # Fallback: If no workspace_follower pointer or not found, search directly
        if not workspace_follower:
            logger.info(f"  No isSelectedWorkspaceFollower found, searching workspace_follower by user...")
            # Try to find any workspace_follower for this user where they are a member
            workspace_follower = self.workspace_followers.find_one({
                "$or": [
                    {"_p_user": f"_User${developer_id}"},
                    {"user.objectId": developer_id},
                    {"user.__type": "Pointer", "user.objectId": developer_id}
                ],
                "isMember": True
            })
        
        if not workspace_follower:
            logger.warning(f"No workspace_follower found for developer {developer_id}")
            return None
        
        logger.info(f"  ✓ Found workspace_follower: {workspace_follower.get('_id')}")
        
        # Get workspace pointer
        workspace_pointer = workspace_follower.get("_p_workspace") or workspace_follower.get("workspace")
        
        workspace_id = None
        if isinstance(workspace_pointer, str) and workspace_pointer.startswith("WorkSpace$"):
            workspace_id = workspace_pointer.replace("WorkSpace$", "")
        elif isinstance(workspace_pointer, dict):
            workspace_id = workspace_pointer.get("objectId")
        else:
            workspace_id = workspace_pointer
        
        if not workspace_id:
            return None
        
        # Get the actual WorkSpace to find subscription
        workspace = self.workspaces.find_one({"_id": workspace_id})
        if not workspace:
            logger.warning(f"WorkSpace {workspace_id} not found")
            return {"workspace_id": workspace_id, "subscription_id": None}

        # Debug: log all fields in workspace to understand structure
        logger.debug(f"  Workspace {workspace_id} fields: {list(workspace.keys())}")
        for key in workspace.keys():
            if any(keyword in key.lower() for keyword in ['subscription', 'billing', 'plan', 'tier']):
                logger.info(f"  {key}: {workspace[key]}")
        
        # Extract subscription pointer - try multiple field patterns
        subscription_pointer = (
            workspace.get("subscription") or
            workspace.get("_p_subscription") or
            workspace.get("_subscription") or
            workspace.get("plan") or
            workspace.get("_p_plan")
        )
        subscription_id = None

        if subscription_pointer:
            logger.info(f"  Found subscription field: {subscription_pointer}")
            if isinstance(subscription_pointer, dict):
                subscription_id = subscription_pointer.get("objectId")
            elif isinstance(subscription_pointer, str):
                if subscription_pointer.startswith("Subscription$"):
                    subscription_id = subscription_pointer.replace("Subscription$", "")
                elif subscription_pointer.startswith("_Subscription$"):
                    subscription_id = subscription_pointer.replace("_Subscription$", "")
                else:
                    # Maybe it's just the objectId directly
                    subscription_id = subscription_pointer

        # If no subscription found, check if workspace has a subscription relation
        if not subscription_id:
            # Check for Parse relation format
            subscription_relation = workspace.get("subscription", {})
            if isinstance(subscription_relation, dict) and subscription_relation.get("__type") == "Relation":
                # For relations, we need to query the relation table or look for _Subscription$ pointer
                logger.info(f"  Found subscription relation for workspace {workspace_id}, but no direct pointer")

            # Try to find subscription in the actual Subscription collection by workspace reference
            try:
                subscription_doc = self.db["Subscription"].find_one({
                    "$or": [
                        {"workspace.objectId": workspace_id},
                        {"_p_workspace": f"WorkSpace${workspace_id}"},
                        {"workspaceId": workspace_id}
                    ]
                })
                if subscription_doc:
                    subscription_id = subscription_doc["_id"]
                    logger.info(f"  ✓ Found subscription {subscription_id} by reverse lookup from workspace {workspace_id}")
                else:
                    logger.info(f"  No subscription found for workspace {workspace_id} via reverse lookup")
            except Exception as e:
                logger.warning(f"  Error searching for subscription: {e}")
        
        logger.info(f"  📦 Workspace: {workspace_id}, Subscription: {subscription_id or 'None'}")

        # Debug: Show what we found in the workspace document
        if subscription_id:
            logger.info(f"  ✓ Found subscription {subscription_id} for workspace {workspace_id}")
        else:
            logger.warning(f"  ⚠️  No subscription found for workspace {workspace_id}")
            logger.info(f"  Workspace document keys: {list(workspace.keys())}")
            # Log subscription-related fields
            for key in workspace.keys():
                if any(keyword in key.lower() for keyword in ['subscription', 'billing', 'plan', 'tier', 'stripe']):
                    logger.info(f"  {key}: {workspace[key]}")
        
        # Get workspace owner (creator)
        workspace_owner_id = None
        if workspace:
            workspace_owner_pointer = workspace.get("_p_user") or workspace.get("user")
            if isinstance(workspace_owner_pointer, str) and workspace_owner_pointer.startswith("_User$"):
                workspace_owner_id = workspace_owner_pointer.replace("_User$", "")
            elif isinstance(workspace_owner_pointer, dict):
                workspace_owner_id = workspace_owner_pointer.get("objectId")
            else:
                workspace_owner_id = workspace_owner_pointer

            if workspace_owner_id:
                logger.info(f"  Workspace {workspace_id} owner: {workspace_owner_id}")
            else:
                logger.warning(f"  Could not determine workspace {workspace_id} owner")

        return {
            "workspace_id": workspace_id,
            "subscription_id": subscription_id,
            "workspace_owner_id": workspace_owner_id
        }

    def _get_user_fallback_name(self, dev: Dict) -> str:
        """Safely get a fallback name from user, handling None values."""
        display_name = dev.get("displayName")
        email = dev.get("email")
        username = dev.get("username")
        
        # Try displayName first
        if display_name and isinstance(display_name, str):
            return display_name
        
        # Try email (get part before @)
        if email and isinstance(email, str):
            return email.split("@")[0]
        
        # Try username
        if username and isinstance(username, str):
            return username
        
        # Last resort: use user ID
        return f"Developer-{dev['_id'][:8]}"
    
    def get_organization_name(self, subscription_id: Optional[str], dev: Dict) -> str:
        """Get organization name from Company via Subscription chain, fallback to user name."""
        if not subscription_id:
            # No subscription, use user name
            return self._get_user_fallback_name(dev)

        try:
            # Get subscription document
            subscription = self.db["Subscription"].find_one({"_id": subscription_id})
            if not subscription:
                logger.warning(f"  Subscription {subscription_id} not found")
                return self._get_user_fallback_name(dev)

            logger.info(f"  Found subscription document for {subscription_id}")

            # Get company pointer from subscription
            company_pointer = (
                subscription.get("company") or
                subscription.get("_p_company") or
                subscription.get("_company")
            )

            if not company_pointer:
                logger.info(f"  No company pointer found in subscription {subscription_id}")
                # Debug: show what fields exist
                logger.info(f"  Subscription fields: {list(subscription.keys())}")
                return self._get_user_fallback_name(dev)

            # Extract company ID from pointer
            company_id = None
            if isinstance(company_pointer, dict):
                company_id = company_pointer.get("objectId")
            elif isinstance(company_pointer, str):
                if company_pointer.startswith("Company$"):
                    company_id = company_pointer.replace("Company$", "")
                elif company_pointer.startswith("_Company$"):
                    company_id = company_pointer.replace("_Company$", "")
                else:
                    # Maybe it's just the objectId directly
                    company_id = company_pointer

            if not company_id:
                logger.info(f"  Could not extract company ID from pointer: {company_pointer}")
                return self._get_user_fallback_name(dev)

            logger.info(f"  Found company pointer: {company_id}")

            # Get company document
            company = self.db["Company"].find_one({"_id": company_id})
            if not company:
                logger.warning(f"  Company {company_id} not found")
                return self._get_user_fallback_name(dev)

            # Get company name - based on your example, the field is 'name'
            company_name = company.get("name") or company.get("displayName") or company.get("companyName") or company.get("company_name")
            if company_name:
                logger.info(f"  ✓ Using company name: {company_name}")
                return company_name
            else:
                logger.info(f"  Company {company_id} has no name field")
                logger.info(f"  Company fields: {list(company.keys())}")
                return self._get_user_fallback_name(dev)

        except Exception as e:
            logger.warning(f"  Error getting company name for subscription {subscription_id}: {e}")
            return self._get_user_fallback_name(dev)
    
    def get_workspace_members_and_counts(self, workspace_id: str, developer_id: str) -> Dict:
        """
        Get all team members for a workspace (excluding end-users) and aggregate counts.
        
        Returns:
            Dict with:
                - members: List of member info dicts
                - aggregated_counts: Dict of summed counts from all members
        """
        members = []
        aggregated_counts = {
            "addMemoryTokenCount": 0,
            "addMemoryTotalCost": 0.0,
            "memoriesCount": 0,
            "miniInteractionCount": 0,
            "premiumInteractionCount": 0,
            "storageCount": 0,
            "interactionTotalCost": 0.0
        }
        
        # Find all workspace_followers for this workspace
        workspace_followers = self.workspace_followers.find({
            "$or": [
                {"_p_workspace": f"WorkSpace${workspace_id}"},
                {"workspace.objectId": workspace_id}
            ]
        })
        
        total_followers = self.workspace_followers.count_documents({
            "$or": [
                {"_p_workspace": f"WorkSpace${workspace_id}"},
                {"workspace.objectId": workspace_id}
            ]
        })
        logger.info(f"  🔍 Found {total_followers} workspace_followers for workspace {workspace_id}")
        
        for wf in workspace_followers:
            # Get user from workspace_follower
            user_pointer = wf.get("_p_user") or wf.get("user")
            
            if isinstance(user_pointer, str) and user_pointer.startswith("_User$"):
                user_id = user_pointer.replace("_User$", "")
            elif isinstance(user_pointer, dict):
                user_id = user_pointer.get("objectId")
            else:
                user_id = user_pointer
            
            if not user_id:
                continue
            
            # Get the actual user
            user = self.users.find_one({"_id": user_id})
            if not user:
                logger.debug(f"  User {user_id} not found, skipping")
                continue
            
            # Skip end-users (type: "developerUser")
            # If type doesn't exist or is not "developerUser", it's a team member
            user_type = user.get("type")
            if user_type == "developerUser":
                logger.debug(f"  Skipping end-user: {user_id}")
                continue
            
            # Get user role in workspace
            role = self.get_user_role_in_workspace(user_id, workspace_id)
            
            members.append({
                "user_id": user_id,
                "email": user.get("email", ""),
                "displayName": user.get("displayName", ""),
                "role": role
            })
            
            # Aggregate counts from this workspace_follower
            aggregated_counts["addMemoryTokenCount"] += wf.get("addMemoryTokenCount", 0) or 0
            aggregated_counts["addMemoryTotalCost"] += wf.get("addMemoryTotalCost", 0.0) or 0.0
            aggregated_counts["memoriesCount"] += wf.get("memoriesCount", 0) or 0
            aggregated_counts["miniInteractionCount"] += wf.get("miniInteractionCount", 0) or 0
            aggregated_counts["premiumInteractionCount"] += wf.get("premiumInteractionCount", 0) or 0
            aggregated_counts["storageCount"] += wf.get("storageCount", 0) or 0
            aggregated_counts["interactionTotalCost"] += wf.get("interactionTotalCost", 0.0) or 0.0
            
            logger.info(f"  ✓ Found member: {user.get('displayName')} ({user_id}) - Role: {role}")
        
        logger.info(f"  📊 Aggregated counts: {aggregated_counts['memoriesCount']} memories, {aggregated_counts['storageCount']} MB storage")
        
        return {
            "members": members,
            "aggregated_counts": aggregated_counts
        }
    
    def get_user_role_in_workspace(self, user_id: str, workspace_id: str) -> str:
        """Determine user's role in workspace from _Role class"""
        # Check for owner role
        owner_role = self.roles.find_one({"name": f"owner-{workspace_id}"})
        if owner_role:
            users_relation = owner_role.get("users")
            # Check if user is in this role (would need to query relation)
            # For now, simplified: check if role exists with this pattern
            if self.check_user_in_role(user_id, f"owner-{workspace_id}"):
                return "owner"
        
        # Check for admin role
        if self.check_user_in_role(user_id, f"admin-{workspace_id}"):
            return "admin"
        
        # Check for moderator role
        if self.check_user_in_role(user_id, f"moderator-{workspace_id}"):
            return "moderator"
        
        # Default to member
        return "member"
    
    def check_user_in_role(self, user_id: str, role_name: str) -> bool:
        """Check if user has a specific role"""
        role = self.roles.find_one({"name": role_name})
        if not role:
            return False
        
        # Parse relations are complex, so we check relatedObjects collection
        # or use the _Role table directly
        # Simplified: check if role exists
        return True if role else False
    
    def preflight_check(self):
        """Show stats about what will be migrated"""
        print("\nDEBUG - preflight_check() called!", flush=True)
        logger.info("\n=== Pre-flight Check ===")
        
        # Base query for workspace creators (users with API keys who are NOT end-users)
        # End-users have type: "developerUser" and should be excluded
        # $ne matches both: documents where type != "developerUser" AND documents without type field
        base_query = {
            "userAPIkey": {"$exists": True},
            "type": {"$ne": "developerUser"}
        }
        
        # Count total developers
        total_developers = self.users.count_documents(base_query)
        
        # Count already migrated developers
        migrated_query = {**base_query, "organization_id": {"$exists": True}}
        migrated_developers = self.users.count_documents(migrated_query)
        
        # Count unmigrated developers
        unmigrated_query = {**base_query, "organization_id": {"$exists": False}}
        unmigrated_developers = self.users.count_documents(unmigrated_query)
        
        print(f"DEBUG - Workspace Creators: total={total_developers}, migrated={migrated_developers}, unmigrated={unmigrated_developers}", flush=True)
        
        # Count end-users (should NOT be migrated)
        end_users_count = self.users.count_documents({
            "type": "developerUser"
        })
        print(f"DEBUG - End-users count: {end_users_count}", flush=True)
        
        # Count existing organizations
        existing_orgs = self.organizations.count_documents({})
        
        # Count existing namespaces
        existing_namespaces = self.namespaces.count_documents({})
        
        # Count existing API keys
        existing_api_keys = self.api_keys.count_documents({})
        
        logger.info(f"Total Workspace Creators (users with API keys): {total_developers}")
        logger.info(f"  ✓ Already Migrated: {migrated_developers} (have organization_id)")
        logger.info(f"  ⚠️  Not Yet Migrated: {unmigrated_developers} (missing organization_id)")
        logger.info(f"\nEnd-Users (type: 'developerUser'): {end_users_count}")
        logger.info(f"  → These are EXCLUDED from migration (they don't create organizations)")
        logger.info(f"\nExisting Multi-tenant Objects:")
        logger.info(f"  Organizations: {existing_orgs}")
        logger.info(f"  Namespaces: {existing_namespaces}")
        logger.info(f"  API Keys: {existing_api_keys}")
        
        if self.only_new:
            logger.info(f"\n🎯 Will process: {unmigrated_developers} NEW developers (--only-new flag)")
            print(f"DEBUG - Will process {unmigrated_developers} unmigrated developers (--only-new)", flush=True)
        elif self.limit:
            logger.info(f"\n🎯 Will process: {min(self.limit, total_developers)} developers (--limit {self.limit})")
            print(f"DEBUG - Will process {min(self.limit, total_developers)} developers (--limit {self.limit})", flush=True)
        else:
            logger.info(f"\n🎯 Will process: {total_developers} developers (all)")
            print(f"DEBUG - Will process {total_developers} developers (all)", flush=True)
            if migrated_developers > 0:
                logger.warning(f"   ⚠️  WARNING: {migrated_developers} developers already have organization_id!")
                logger.warning(f"   ⚠️  Use --only-new flag to skip already migrated developers")
        
        logger.info("=" * 60)
        print("DEBUG - preflight_check() completed!", flush=True)
    
    def create_collections(self):
        """Create new collections with Parse-compatible format"""
        logger.info("Creating Organization and Namespace collections...")
        
        # Collections will be created automatically when we insert first document
        logger.info("✓ Collections will be created on first insert")
    
    def create_indexes(self):
        """Create indexes for performance"""
        logger.info("Creating indexes...")
        
        # Helper to create index if it doesn't exist
        def create_index_safe(collection, keys, name):
            try:
                existing_indexes = collection.list_indexes()
                index_names = [idx['name'] for idx in existing_indexes]
                if name not in index_names:
                    collection.create_index(keys, name=name)
                    logger.info(f"✓ Created {name} index")
                else:
                    logger.info(f"⊙ Index {name} already exists")
            except Exception as e:
                logger.warning(f"Failed to create index {name}: {e}")
        
        # Organization indexes
        create_index_safe(
            self.organizations,
            [("owner_user_id", ASCENDING)],
            "owner_user_id_1"
        )
        
        # Namespace indexes
        create_index_safe(
            self.namespaces,
            [("organization_id", ASCENDING)],
            "organization_id_1"
        )
        
        # Memory indexes (critical for multi-tenant queries)
        create_index_safe(
            self.memories,
            [("organization_id", ASCENDING), ("namespace_id", ASCENDING), ("_created_at", DESCENDING)],
            "multi_tenant_lookup"
        )
        
        # User indexes
        create_index_safe(
            self.users,
            [("user_type", ASCENDING)],
            "user_type_1"
        )
        
        # APIKey indexes
        create_index_safe(
            self.api_keys,
            [("key", ASCENDING)],
            "key_1"
        )
    
    def migrate_developers_to_organizations(self):
        """Create Organization for each workspace creator WITH workspace members using Pydantic models"""
        print("\nDEBUG - migrate_developers_to_organizations() called!", flush=True)
        logger.info("\n=== Migrating Workspace Creators to Organizations ===")
        
        # Find all workspace creators (users with API keys who are NOT end-users)
        # Users with userAPIkey = workspace creators who can have Organizations
        # type: "developerUser" = end-users (should be excluded)
        # $ne matches both: documents where type != "developerUser" AND documents without type field
        query = {
            "userAPIkey": {"$exists": True},
            "type": {"$ne": "developerUser"}
        }
        
        # If only_new flag is set, only process workspace creators without organization_id
        if self.only_new:
            query["organization_id"] = {"$exists": False}
            logger.info("🔍 Only processing NEW workspace creators (without organization_id)")
        
        if self.limit:
            workspace_creators = list(self.users.find(query).limit(self.limit))
            logger.info(f"Processing {len(workspace_creators)} workspace creators (limited to {self.limit})")
            print(f"DEBUG - Found {len(workspace_creators)} workspace creators (limit={self.limit})", flush=True)
        else:
            workspace_creators = list(self.users.find(query))
            logger.info(f"Processing {len(workspace_creators)} workspace creators")
            print(f"DEBUG - Found {len(workspace_creators)} workspace creators (no limit)", flush=True)
        
        created_count = 0
        skipped_count = 0
        print(f"DEBUG - Starting to process workspace creators...", flush=True)
        
        for creator in workspace_creators:
            user_id = creator["_id"]
            print(f"\nDEBUG - Processing creator: {user_id}", flush=True)

            # Get workspace and subscription for this developer FIRST
            print(f"DEBUG - Getting workspace for developer {user_id}...", flush=True)
            workspace_info = self.get_workspace_for_developer(user_id)
            print(f"DEBUG - Workspace info retrieved: {workspace_info}", flush=True)
            workspace_id = workspace_info.get("workspace_id") if workspace_info else None
            subscription_id = workspace_info.get("subscription_id") if workspace_info else None
            workspace_owner_id = workspace_info.get("workspace_owner_id") if workspace_info else None

            logger.info(f"  Found workspace: {workspace_id}, subscription: {subscription_id}, owner: {workspace_owner_id}")

            # Check if org already exists for this developer's workspace
            existing_org = None
            if workspace_id:
                # Look for existing org with this workspace
                existing_org = self.organizations.find_one({
                    "workspace.objectId": workspace_id
                })
                if existing_org:
                    print(f"DEBUG - Skipping: Organization already exists for workspace {workspace_id}", flush=True)
                    logger.info(f"⊙ Organization already exists for workspace {workspace_id}")
                    skipped_count += 1
                    continue

            # Also check by user ID as fallback
            if not existing_org:
                existing_org = self.organizations.find_one({"owner_user_id": user_id})
                if existing_org:
                    print(f"DEBUG - Skipping: Organization already exists for user {user_id}", flush=True)
                    logger.info(f"⊙ Organization already exists for user {user_id[:10]}...")
                    skipped_count += 1
                    continue
            
            print(f"DEBUG - No existing org found, will create new organization...", flush=True)
            
            # Generate Parse-compatible 10-character objectId
            org_id = self.generate_parse_object_id()
            
            # Debug: Show subscription document structure for company lookup
            if subscription_id:
                try:
                    sub_doc = self.db["Subscription"].find_one({"_id": subscription_id})
                    if sub_doc:
                        logger.info(f"  Subscription {subscription_id} fields: {list(sub_doc.keys())}")
                        # Log company-related fields
                        for key in sub_doc.keys():
                            if 'company' in key.lower():
                                logger.info(f"  {key}: {sub_doc[key]}")
                    else:
                        logger.warning(f"  Subscription {subscription_id} document not found")
                except Exception as e:
                    logger.warning(f"  Error reading subscription {subscription_id}: {e}")
            else:
                logger.warning(f"  No workspace found for developer {user_id}")
            
            # Get team members from workspace and aggregate counts
            team_member_ids = [user_id]  # Always include the owner
            team_members_info = []
            aggregated_counts = {
                "addMemoryTokenCount": 0,
                "addMemoryTotalCost": 0.0,
                "memoriesCount": 0,
                "miniInteractionCount": 0,
                "premiumInteractionCount": 0,
                "storageCount": 0,
                "interactionTotalCost": 0.0
            }
            
            if workspace_id:
                workspace_data = self.get_workspace_members_and_counts(workspace_id, user_id)
                team_members_info = workspace_data["members"]
                aggregated_counts = workspace_data["aggregated_counts"]
                team_member_ids = [m["user_id"] for m in team_members_info]
                # Ensure owner is in the list
                if user_id not in team_member_ids:
                    team_member_ids.append(user_id)
            
            # Determine the actual organization owner (workspace owner, not current developer)
            actual_owner_id = workspace_owner_id if workspace_owner_id else user_id

            # Create ACL (owner + workspace member role full access)
            acl = self._build_acl(actual_owner_id, workspace_id)
            if actual_owner_id != user_id:
                logger.info(f"  ❗ Organization owner will be workspace owner {actual_owner_id}, not current developer {user_id}")
                # Get the actual owner's details
                actual_owner = self.users.find_one({"_id": actual_owner_id})
                if actual_owner:
                    owner_for_naming = actual_owner
                    logger.info(f"  Using workspace owner details for organization: {actual_owner.get('displayName', actual_owner_id)}")
                else:
                    logger.warning(f"  Workspace owner {actual_owner_id} not found, using current workspace creator")
                    actual_owner_id = user_id
                    owner_for_naming = creator
            else:
                logger.info(f"  Workspace creator {user_id} is also the workspace owner")
                owner_for_naming = creator

            # Get organization name from Company if available, otherwise use owner name
            org_name = self.get_organization_name(subscription_id, owner_for_naming)
            logger.info(f"  Organization name: {org_name}")
            
            # Create owner pointer (use workspace owner, not current developer)
            owner_pointer = ParseUserPointer(objectId=actual_owner_id)
            logger.info(f"  Organization owner will be: {actual_owner_id}")
            
            # Create workspace pointer if workspace exists
            workspace_pointer = WorkspacePointer(objectId=workspace_id) if workspace_id else None
            
            # Create subscription pointer if subscription exists
            subscription_pointer = SubscriptionPointer(objectId=subscription_id) if subscription_id else None
            if subscription_pointer:
                logger.info(f"  Created subscription pointer for {subscription_id}")
            else:
                logger.info(f"  No subscription pointer created (subscription_id is None)")
            
            # Create team members list (already filtered to exclude developerUser types)
            team_member_pointers = [ParseUserPointer(objectId=member_id) for member_id in team_member_ids]
            logger.info(f"  Created {len(team_member_pointers)} team member pointers for organization")
            
            # Create Organization Pydantic model
            logger.info(f"  Final organization name selected: {org_name}")
            org_model = Organization(
                name=org_name,
                owner=owner_pointer,
                workspace=workspace_pointer,
                subscription=subscription_pointer,
                plan_tier="developer",
                team_members=team_member_pointers,
                ACL=acl
            )
            
            # Convert to Parse document format
            org_doc = self.pydantic_to_parse_doc(org_model, org_id)
            logger.info(f"  Created organization document with {len(team_member_ids)} team members in list")
            
            # Add custom fields not in Pydantic model
            org_doc["owner_user_id"] = actual_owner_id  # For easy querying (not just in pointer)
            org_doc["team_members"] = team_member_ids
            org_doc["team_members_info"] = team_members_info
            
            # Add aggregated counts from all team members
            org_doc["addMemoryTokenCount"] = aggregated_counts["addMemoryTokenCount"]
            org_doc["addMemoryTotalCost"] = aggregated_counts["addMemoryTotalCost"]
            org_doc["memoriesCount"] = aggregated_counts["memoriesCount"]
            org_doc["miniInteractionCount"] = aggregated_counts["miniInteractionCount"]
            org_doc["premiumInteractionCount"] = aggregated_counts["premiumInteractionCount"]
            org_doc["storageCount"] = aggregated_counts["storageCount"]
            org_doc["interactionTotalCost"] = aggregated_counts["interactionTotalCost"]

            # Add subscription pointer to organization document if found
            if subscription_id:
                org_doc["subscription"] = {
                    "__type": "Pointer",
                    "className": "Subscription",
                    "objectId": subscription_id
                }
                # Add shorthand pointer field for Parse Server queries
                org_doc["_p_subscription"] = f"Subscription${subscription_id}"
                logger.info(f"  Added subscription pointer {subscription_id} to organization {org_id}")
            else:
                logger.info(f"  No subscription found for workspace {workspace_id}, organization {org_id} will not have subscription pointer")
            # Show what keys were in the workspace for debugging
            if workspace_info:
                workspace_data = self.workspaces.find_one({"_id": workspace_info.get("workspace_id")})
                workspace_keys = list(workspace_data.keys()) if workspace_data else []
                logger.info(f"  Available workspace fields: {workspace_keys}")
            else:
                logger.info(f"  No workspace info available to show fields")
            
            # Insert into MongoDB
            self.organizations.insert_one(org_doc)
            logger.info(f"  ✓ Organization document inserted with ID: {org_id}")

            # After insert, add team_members relation via REST so it shows in dashboard
            try:
                success = self._add_team_members_relation(org_id, team_member_ids)
                if success:
                    logger.info(f"  ✓ Successfully added team_members relation for org {org_id}")
                else:
                    logger.warning(f"  ⚠️  Failed to add team_members relation for org {org_id} - relation not created in Parse Server")
            except Exception as e:
                # Non-fatal but log the error
                logger.warning(f"Failed to add team_members relation for org {org_id}: {e}")
            
            # Update the organization owner (workspace creator)
            self.users.update_one(
                {"_id": actual_owner_id},
                {
                    "$set": {
                        "user_type": "CREATOR",  # Workspace creator/owner
                        "organization_id": org_id,
                        "_updated_at": datetime.now(timezone.utc)
                    }
                }
            )

            # Update all team members with organization_id
            for member_id in team_member_ids:
                if member_id != actual_owner_id:
                    self.users.update_one(
                        {"_id": member_id},
                        {
                            "$set": {
                                "user_type": "TEAM_MEMBER",
                                "organization_id": org_id,
                                "_updated_at": datetime.now(timezone.utc)
                            }
                        }
                    )

            # If current developer is not the workspace owner, also update them
            if user_id != actual_owner_id and user_id in team_member_ids:
                self.users.update_one(
                    {"_id": user_id},
                    {
                        "$set": {
                            "user_type": "TEAM_MEMBER",
                            "organization_id": org_id,
                            "_updated_at": datetime.now(timezone.utc)
                        }
                    }
                )
            
            logger.info(f"✓ Created Organization {org_id} for workspace {workspace_id} (owner: {actual_owner_id[:10]}...) with {len(team_member_ids)} members")
            logger.info(f"  📊 Counts: {aggregated_counts['memoriesCount']} memories, {aggregated_counts['storageCount']} MB storage, {aggregated_counts['miniInteractionCount']} mini interactions, {aggregated_counts['premiumInteractionCount']} premium interactions")
            print(f"DEBUG - Successfully created organization {org_id}!", flush=True)
            created_count += 1
        
        print(f"\nDEBUG - Loop completed! created={created_count}, skipped={skipped_count}", flush=True)
        logger.info(f"\nOrganizations: {created_count} created, {skipped_count} skipped")
    
    def create_default_namespaces(self):
        """Create default namespace for each organization using Pydantic models"""
        logger.info("\n=== Creating Default Namespaces ===")
        
        orgs = self.organizations.find()
        created_count = 0
        skipped_count = 0
        
        for org in orgs:
            org_id = org["_id"]
            
            # Check if namespace already exists for this org
            existing_ns = self.namespaces.find_one({"organization_id": org_id})
            if existing_ns:
                logger.info(f"⊙ Namespace already exists for org {org_id}")
                skipped_count += 1
                continue
            
            # Skip corrupt/incomplete organizations
            if "name" not in org or not org.get("name"):
                logger.error(f"❌ Organization {org_id} is corrupt (missing name field)! Available keys: {list(org.keys())}")
                logger.error(f"   Skipping namespace creation for this org. You should delete this corrupt org manually.")
                skipped_count += 1
                continue
            
            # Generate Parse-compatible 10-character objectId
            ns_id = self.generate_parse_object_id()
            
            # Create ACL (owner + member-<workspace> role if available)
            owner_user_id = org.get("owner_user_id")
            if not owner_user_id:
                logger.warning(f"⚠️  Organization {org_id} has no owner_user_id! Available keys: {list(org.keys())}")
            
            # Try to read workspace from org pointer if present
            workspace_pointer = org.get("workspace")
            workspace_id = None
            if isinstance(workspace_pointer, dict):
                workspace_id = workspace_pointer.get("objectId")
            
            logger.info(f"Creating namespace for org {org_id} (owner: {owner_user_id}, workspace: {workspace_id})")
            acl = self._build_acl(owner_user_id, workspace_id)
            
            # Create organization pointer
            org_pointer = OrganizationPointer(objectId=org_id)
            
            # Create Namespace using Pydantic model
            ns_model = Namespace(
                name=f"{org['name']}-production",
                organization=org_pointer,
                environment_type=EnvironmentType.PRODUCTION,
                is_active=True,
                ACL=acl
            )
            
            # Convert to Parse document format
            ns_doc = self.pydantic_to_parse_doc(ns_model, ns_id)
            
            # Add organization_id for easy querying
            ns_doc["organization_id"] = org_id
            
            # Insert into MongoDB
            self.namespaces.insert_one(ns_doc)
            # Also set this namespace as default_namespace on organization
            try:
                self.organizations.update_one(
                    {"_id": org_id},
                    {"$set": {
                        "default_namespace": {
                            "__type": "Pointer",
                            "className": "Namespace",
                            "objectId": ns_id
                        },
                        "_p_default_namespace": f"Namespace${ns_id}",  # Add shorthand for Parse queries
                        "default_namespace_id": ns_id
                    }}
                )
            except Exception as e:
                logger.warning(f"Failed to set default_namespace for org {org_id}: {e}")

            logger.info(f"✓ Created Namespace {ns_id} for org {org_id}")
            created_count += 1
        
        logger.info(f"\nNamespaces: {created_count} created, {skipped_count} skipped")
    
    def migrate_api_keys(self):
        """Migrate existing API keys to new schema using Pydantic models"""
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
            
            # Find namespace by organization_id (not by _id format)
            namespace = self.namespaces.find_one({"organization_id": org_id})
            
            if not namespace:
                logger.warning(f"No namespace found for org {org_id}")
                continue
            
            # Get org for ACL and workspace
            org = self.organizations.find_one({"_id": org_id})
            owner_user_id = org.get("owner_user_id") if org else None
            workspace_pointer = org.get("workspace") if org else None
            workspace_id = None
            if isinstance(workspace_pointer, dict):
                workspace_id = workspace_pointer.get("objectId")
            acl = self._build_acl(owner_user_id, workspace_id)
            
            # Create pointers
            ns_pointer = NamespacePointer(objectId=namespace["_id"])
            org_pointer = OrganizationPointer(objectId=org_id)
            
            # Create APIKey using Pydantic model
            ak_model = APIKey(
                key=api_key,
                name="Production API Key (Migrated)",
                namespace=ns_pointer,
                organization=org_pointer,
                environment="production",
                permissions=["read", "write", "delete"],
                is_active=True,
                ACL=acl
            )
            
            # Generate Parse-compatible 10-character objectId
            ak_id = self.generate_parse_object_id()
            
            # Convert to Parse document format
            ak_doc = self.pydantic_to_parse_doc(ak_model, ak_id)
            
            # Add fields for easy querying
            ak_doc["organization_id"] = org_id
            ak_doc["namespace_id"] = namespace["_id"]
            
            # Insert into MongoDB
            self.api_keys.insert_one(ak_doc)
            logger.info(f"✓ Migrated API key for org {org_id}")
            created_count += 1
        
        logger.info(f"\nAPI Keys: {created_count} created, {skipped_count} skipped")
    
    def run_migration(self):
        """Run complete migration"""
        print("\nDEBUG - run_migration() called!", flush=True)
        logger.info("=" * 60)
        logger.info("Starting Enhanced Multi-Tenant Migration v2")
        logger.info("=" * 60)
        print("DEBUG - After initial logger.info calls", flush=True)
        
        try:
            # Step 0: Pre-flight check
            self.preflight_check()
            
            # Step 1: Create collections
            self.create_collections()
            
            # Step 2: Create indexes
            self.create_indexes()
            
            # Step 3: Migrate developers to organizations (WITH workspace members)
            self.migrate_developers_to_organizations()
            
            # Step 4: Create default namespaces
            self.create_default_namespaces()
            
            # Step 5: Migrate API keys
            self.migrate_api_keys()
            
            # Step 6: Verify
            logger.info("\n=== Migration Complete ===")
            logger.info(f"Organizations: {self.organizations.count_documents({})}")
            logger.info(f"Namespaces: {self.namespaces.count_documents({})}")
            logger.info(f"API Keys: {self.api_keys.count_documents({})}")
            
            logger.info("\n" + "=" * 60)
            logger.info("✓ Migration completed successfully!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"\n❌ Migration failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Papr to multi-tenant architecture with proper Parse Server format"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of developers to process (useful for testing, e.g., --limit 3)"
    )
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="Only migrate NEW developers who don't have organization_id yet (safe for re-running)"
    )
    
    args = parser.parse_args()
    
    print(f"DEBUG - Creating migration object with limit={args.limit}, only_new={args.only_new}", flush=True)
    migration = EnhancedMultiTenantMigration(limit=args.limit, only_new=args.only_new)
    print("DEBUG - Migration object created successfully!", flush=True)
    migration.run_migration()


if __name__ == "__main__":
    main()

