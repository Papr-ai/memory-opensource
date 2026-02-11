#!/usr/bin/env python3
"""
Bootstrap Open Source User for Papr Memory

This script creates a complete user setup for Papr Memory open source:
1. Creates Parse _User (without Auth0)
2. Creates Organization
3. Creates default Namespace
4. Generates API key
5. Links everything together

This replaces the cloud signup flow (Auth0 + dashboard) for open source users.

Usage:
    python scripts/bootstrap_opensource_user.py \
        --email developer@example.com \
        --name "Developer Name" \
        --organization "My Company"

    # With custom password
    python scripts/bootstrap_opensource_user.py \
        --email dev@example.com \
        --name "Dev" \
        --organization "My Org" \
        --password "securepassword123"
"""

import os
import sys
import secrets
import requests
import argparse
from datetime import datetime
from typing import Optional, Dict, Any


def generate_api_key() -> str:
    """Generate a secure API key"""
    return f"pmem_oss_{secrets.token_urlsafe(32)}"


def generate_password() -> str:
    """Generate a secure random password"""
    return secrets.token_urlsafe(16)


class OpenSourceBootstrap:
    def __init__(self, parse_url: str, app_id: str, master_key: str):
        self.parse_url = parse_url
        self.app_id = app_id
        self.master_key = master_key
        self.headers = {
            "X-Parse-Application-Id": app_id,
            "X-Parse-Master-Key": master_key,
            "Content-Type": "application/json"
        }

    def create_user(self, email: str, name: str, password: str, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Create Parse _User with userAPIkey field

        Returns user object with objectId and sessionToken
        """
        print(f"\n📝 Creating Parse User: {email}")

        # Check if user already exists
        check_url = f"{self.parse_url}/parse/users"
        params = {
            "where": f'{{"email": "{email}"}}'
        }
        response = requests.get(check_url, headers=self.headers, params=params, timeout=10)

        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                print(f"   ⚠️  User already exists: {results[0].get('objectId')}")
                return results[0]

        # Create new user with userAPIkey field
        # Matches structure from Auth0 PostLogin action for compatibility
        user_data = {
            "username": email,
            "email": email,
            "password": password,
            "name": name,
            "fullname": name,  # Matches Auth0 action
            "displayName": name.split()[0] if name else email.split('@')[0],  # First name or email prefix
            "emailVerified": True,  # Auto-verify for open source
            "emailVerifiedAuth0": True,  # Compatibility field
            "user_type": "CREATOR",  # Mark as workspace creator
            "userAPIkey": api_key,  # Store API key on user
            "isLogin": True,  # Matches Auth0 action
            "completedProfileSignup": True,  # Auto-complete for CLI users
            "socialProfilePicURL": None,  # No OAuth in open source
            "auth0ID": None,  # No Auth0 in open source
        }

        response = requests.post(
            f"{self.parse_url}/parse/users",
            headers=self.headers,
            json=user_data,
            timeout=10
        )

        if response.status_code in [200, 201]:
            user = response.json()
            print(f"   ✅ User created: {user.get('objectId')}")
            return user
        else:
            print(f"   ❌ Failed to create user: {response.status_code}")
            print(f"   Response: {response.text}")
            return None

    def create_workspace(self, name: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Create WorkSpace for the user

        Returns workspace object with objectId
        """
        print(f"\n💼 Creating Workspace: {name}")

        workspace_data = {
            "name": name,
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            },
            "ACL": {
                user_id: {"read": True, "write": True}
            }
        }

        response = requests.post(
            f"{self.parse_url}/parse/classes/WorkSpace",
            headers=self.headers,
            json=workspace_data,
            timeout=10
        )

        if response.status_code in [200, 201]:
            workspace = response.json()
            print(f"   ✅ Workspace created: {workspace.get('objectId')}")
            return workspace
        else:
            print(f"   ❌ Failed to create workspace: {response.status_code}")
            print(f"   Response: {response.text}")
            return None

    def create_workspace_follower(self, user_id: str, workspace_id: str) -> Optional[Dict[str, Any]]:
        """
        Create workspace_follower relationship

        Returns workspace_follower object with objectId
        """
        print(f"\n🔗 Creating workspace_follower relationship...")

        follower_data = {
            "user": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            },
            "workspace": {
                "__type": "Pointer",
                "className": "WorkSpace",
                "objectId": workspace_id
            },
            "ACL": {
                user_id: {"read": True, "write": True}
            }
        }

        response = requests.post(
            f"{self.parse_url}/parse/classes/workspace_follower",
            headers=self.headers,
            json=follower_data,
            timeout=10
        )

        if response.status_code in [200, 201]:
            follower = response.json()
            print(f"   ✅ workspace_follower created: {follower.get('objectId')}")
            return follower
        else:
            print(f"   ❌ Failed to create workspace_follower: {response.status_code}")
            print(f"   Response: {response.text}")
            return None

    def update_user_workspace_follower(self, user_id: str, follower_id: str) -> bool:
        """Set isSelectedWorkspaceFollower on user"""
        print(f"\n🔗 Setting isSelectedWorkspaceFollower on user...")

        update_data = {
            "isSelectedWorkspaceFollower": {
                "__type": "Pointer",
                "className": "workspace_follower",
                "objectId": follower_id
            }
        }

        response = requests.put(
            f"{self.parse_url}/parse/users/{user_id}",
            headers=self.headers,
            json=update_data,
            timeout=10
        )

        if response.status_code == 200:
            print(f"   ✅ isSelectedWorkspaceFollower set")
            return True
        else:
            print(f"   ⚠️  Failed to set isSelectedWorkspaceFollower: {response.status_code}")
            return False

    def create_organization(self, name: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Create Organization for the user

        Returns organization object with objectId
        """
        print(f"\n🏢 Creating Organization: {name}")

        org_data = {
            "name": name,
            "slug": name.lower().replace(" ", "-"),
            "created_by": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": user_id
            },
            "plan_tier": "FREE",  # Open source tier
            "rate_limits": {
                "requests_per_hour": 1000,
                "memories_per_month": 10000
            },
            "settings": {
                "allow_api_access": True,
                "enable_analytics": False  # No analytics in open source
            }
        }

        response = requests.post(
            f"{self.parse_url}/parse/classes/Organization",
            headers=self.headers,
            json=org_data,
            timeout=10
        )

        if response.status_code in [200, 201]:
            org = response.json()
            print(f"   ✅ Organization created: {org.get('objectId')}")
            return org
        else:
            print(f"   ❌ Failed to create organization: {response.status_code}")
            print(f"   Response: {response.text}")
            return None

    def create_namespace(self, name: str, org_id: str) -> Optional[Dict[str, Any]]:
        """
        Create default Namespace for the organization

        Returns namespace object with objectId
        """
        print(f"\n📁 Creating Namespace: {name}")

        namespace_data = {
            "name": name,
            "slug": name.lower().replace(" ", "-"),
            "organization": {
                "__type": "Pointer",
                "className": "Organization",
                "objectId": org_id
            },
            "environment": "production",
            "settings": {
                "max_memories": 10000,
                "retention_days": 365
            }
        }

        response = requests.post(
            f"{self.parse_url}/parse/classes/Namespace",
            headers=self.headers,
            json=namespace_data,
            timeout=10
        )

        if response.status_code in [200, 201]:
            namespace = response.json()
            print(f"   ✅ Namespace created: {namespace.get('objectId')}")
            return namespace
        else:
            print(f"   ❌ Failed to create namespace: {response.status_code}")
            print(f"   Response: {response.text}")
            return None

    def update_organization_default_namespace(self, org_id: str, namespace_id: str) -> bool:
        """Update organization to set default namespace"""
        print(f"\n🔗 Linking namespace to organization...")

        update_data = {
            "default_namespace": {
                "__type": "Pointer",
                "className": "Namespace",
                "objectId": namespace_id
            }
        }

        response = requests.put(
            f"{self.parse_url}/parse/classes/Organization/{org_id}",
            headers=self.headers,
            json=update_data,
            timeout=10
        )

        if response.status_code == 200:
            print(f"   ✅ Default namespace set")
            return True
        else:
            print(f"   ⚠️  Failed to set default namespace: {response.status_code}")
            return False

    def generate_api_key(self, api_key: str, user_id: str, org_id: str, namespace_id: str, email: str, name: str) -> bool:
        """
        Create API key entry in Parse Server APIKey class

        Returns True if successful
        """
        print(f"\n🔑 Creating APIKey in Parse Server...")

        # Create APIKey object in Parse Server using actual schema
        # Based on /Users/shawkatkabbara/Downloads/APIKey-2.json
        key_data = {
            "key": api_key,  # The actual API key string
            "name": f"{name} - Default Key",
            "namespace": {
                "__type": "Pointer",
                "className": "Namespace",
                "objectId": namespace_id
            },
            "namespace_id": namespace_id,  # Helper field for easy querying
            "organization": {
                "__type": "Pointer",
                "className": "Organization",
                "objectId": org_id
            },
            "environment": "production",
            "permissions": ["read", "write", "delete"],
            "is_active": True,
            "last_used_at": None,
            "memoriesCount": 0,
            "storageCount": 0,
            # ACL for open source - public read (for API key validation), org write
            "ACL": {
                "*": {"read": True},  # Public read for API key lookup
                org_id: {"read": True, "write": True}  # Organization can manage
            }
        }

        response = requests.post(
            f"{self.parse_url}/parse/classes/APIKey",
            headers=self.headers,
            json=key_data,
            timeout=10
        )

        if response.status_code in [200, 201]:
            print(f"   ✅ APIKey created in APIKey class")
            return True
        else:
            print(f"   ❌ Failed to create APIKey: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    def update_user_organization(self, user_id: str, org_id: str) -> bool:
        """Link user to organization"""
        print(f"\n🔗 Linking user to organization...")

        update_data = {
            "organization": {
                "__type": "Pointer",
                "className": "Organization",
                "objectId": org_id
            }
        }

        response = requests.put(
            f"{self.parse_url}/parse/users/{user_id}",
            headers=self.headers,
            json=update_data,
            timeout=10
        )

        if response.status_code == 200:
            print(f"   ✅ User linked to organization")
            return True
        else:
            print(f"   ⚠️  Failed to link user: {response.status_code}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap Papr Memory open source user",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic setup
  python scripts/bootstrap_opensource_user.py \\
      --email developer@example.com \\
      --name "Developer Name" \\
      --organization "My Company"

  # With custom password
  python scripts/bootstrap_opensource_user.py \\
      --email dev@example.com \\
      --name "Dev" \\
      --organization "My Org" \\
      --password "securepassword123"
        """
    )

    parser.add_argument("--email", help="User email address (default: test@papr.dev)")
    parser.add_argument("--name", help="User full name (default: Test User)")
    parser.add_argument("--organization", help="Organization name (default: Test Organization)")
    parser.add_argument("--password", help="Password (auto-generated if not provided)")
    parser.add_argument("--api-key", help="API key (auto-generated if not provided)")
    parser.add_argument("--parse-url", default=os.getenv("PARSE_SERVER_URL", "http://localhost:1337/parse"),
                       help="Parse Server URL")
    parser.add_argument("--app-id", default=os.getenv("PARSE_APPLICATION_ID") or os.getenv("PARSE_SERVER_APPLICATION_ID"),
                       help="Parse Application ID")
    parser.add_argument("--master-key", default=os.getenv("PARSE_MASTER_KEY") or os.getenv("PARSE_SERVER_MASTER_KEY"),
                       help="Parse Master Key")

    args = parser.parse_args()

    # Use defaults if not provided (for Docker auto-bootstrap)
    if not args.email:
        args.email = "test@papr.dev"
    if not args.name:
        args.name = "Test User"
    if not args.organization:
        args.organization = "Test Organization"

    # Load environment variables
    env_file = ".env.opensource" if os.path.exists(".env.opensource") else ".env"
    if os.path.exists(env_file):
        print(f"📁 Loading environment from {env_file}")
        from dotenv import load_dotenv
        load_dotenv(env_file)

        # Reload args with env vars if not provided
        if not args.app_id:
            args.app_id = os.getenv("PARSE_APPLICATION_ID") or os.getenv("PARSE_SERVER_APPLICATION_ID")
        if not args.master_key:
            args.master_key = os.getenv("PARSE_MASTER_KEY") or os.getenv("PARSE_SERVER_MASTER_KEY")

    if not args.app_id or not args.master_key:
        print("❌ Error: Parse credentials not found")
        print("\nSet in environment or .env file:")
        print("  PARSE_APPLICATION_ID=your-app-id")
        print("  PARSE_MASTER_KEY=your-master-key")
        sys.exit(1)

    # Generate password if not provided
    password = args.password or generate_password()
    password_was_generated = not args.password

    # Generate API key if not provided
    api_key = args.api_key or generate_api_key()
    api_key_was_generated = not args.api_key

    print("\n" + "="*70)
    print("🚀 Papr Memory Open Source - User Bootstrap")
    print("="*70)
    print(f"\n📧 Email: {args.email}")
    print(f"👤 Name: {args.name}")
    print(f"🏢 Organization: {args.organization}")
    print(f"🔐 Password: {'[Auto-generated]' if password_was_generated else '[Custom]'}")
    print(f"🔑 API Key: {'[Auto-generated]' if api_key_was_generated else '[Provided]'}")
    print(f"\n🔗 Parse Server: {args.parse_url}")

    bootstrap = OpenSourceBootstrap(args.parse_url, args.app_id, args.master_key)

    try:
        # Step 1: Create user (with API key)
        user = bootstrap.create_user(args.email, args.name, password, api_key)
        if not user:
            sys.exit(1)

        user_id = user.get("objectId")

        # Step 2: Create workspace
        workspace = bootstrap.create_workspace(f"{args.organization} Workspace", user_id)
        if not workspace:
            sys.exit(1)

        workspace_id = workspace.get("objectId")

        # Step 3: Create workspace_follower
        follower = bootstrap.create_workspace_follower(user_id, workspace_id)
        if not follower:
            sys.exit(1)

        follower_id = follower.get("objectId")

        # Step 4: Set isSelectedWorkspaceFollower on user
        bootstrap.update_user_workspace_follower(user_id, follower_id)

        # Step 5: Create organization (optional, for compatibility)
        org = bootstrap.create_organization(args.organization, user_id)
        if org:
            org_id = org.get("objectId")

            # Step 6: Create namespace
            namespace_name = f"{args.organization} - Default"
            namespace = bootstrap.create_namespace(namespace_name, org_id)
            if namespace:
                namespace_id = namespace.get("objectId")
                # Link namespace to organization
                bootstrap.update_organization_default_namespace(org_id, namespace_id)
                # Link user to organization
                bootstrap.update_user_organization(user_id, org_id)

                # Step 7: Create APIKey entry in Parse Server APIKey class
                api_key_created = bootstrap.generate_api_key(api_key, user_id, org_id, namespace_id, args.email, args.name)
                if not api_key_created:
                    print("   ⚠️  Warning: API key not created in APIKey class, but userAPIkey still set on _User")

        # Update .env file with test credentials
        env_file = ".env.opensource" if os.path.exists(".env.opensource") else ".env"
        if os.path.exists(env_file):
            print(f"\n📝 Updating {env_file} with test credentials...")
            
            # Read existing .env content
            with open(env_file, 'r') as f:
                env_content = f.read()
            
            # Update test environment variables
            updates = {
                'TEST_X_USER_API_KEY': api_key,
                'TEST_USER_ID': user_id,
                'TEST_TENANT_ID': workspace_id,
                'TEST_WORKSPACE_ID': workspace_id,
                'TEST_NAMESPACE_ID': namespace_id if namespace else '',
                'TEST_ORGANIZATION_ID': org_id if org else '',
            }
            
            for key, value in updates.items():
                # Check if the key exists in the file
                if f'{key}=' in env_content:
                    # Replace existing value
                    import re
                    env_content = re.sub(
                        f'^{key}=.*$',
                        f'{key}={value}',
                        env_content,
                        flags=re.MULTILINE
                    )
                else:
                    # Append new key-value pair
                    env_content += f'\n{key}={value}'
            
            # Write back to .env file
            with open(env_file, 'w') as f:
                f.write(env_content)
            
            print(f"   ✅ Updated {env_file} with test credentials")

        # Success!
        print("\n" + "="*70)
        print("✅ Bootstrap Complete!")
        print("="*70)
        print(f"\n📧 Email: {args.email}")
        if password_was_generated:
            print(f"🔐 Password: {password}")
            print("   ⚠️  SAVE THIS PASSWORD - You won't see it again!")
        print(f"\n🔑 API Key: {api_key}")
        print("   ⚠️  SAVE THIS API KEY - You won't see it again!")

        print(f"\n📋 Test credentials saved to {env_file}:")
        print(f"   TEST_X_USER_API_KEY={api_key}")
        print(f"   TEST_USER_ID={user_id}")
        print(f"   TEST_WORKSPACE_ID={workspace_id}")
        if namespace:
            print(f"   TEST_NAMESPACE_ID={namespace_id}")
        if org:
            print(f"   TEST_ORGANIZATION_ID={org_id}")

        print(f"\n🧪 Test your setup:")
        print(f"   curl -X POST http://localhost:5001/v1/memory \\")
        print(f"        -H 'X-API-Key: {api_key}' \\")
        print(f"        -H 'Content-Type: application/json' \\")
        print(f"        -d '{{\"content\": \"My first memory!\"}}'")

        print(f"\n🎯 Login to Parse Dashboard:")
        print(f"   http://localhost:4040")
        print(f"   Username: admin")
        print(f"   Password: password")

        print("\n" + "="*70)

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error: {e}")
        print(f"\nMake sure Parse Server is running at: {args.parse_url}")
        sys.exit(1)


if __name__ == "__main__":
    main()
