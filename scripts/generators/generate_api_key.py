#!/usr/bin/env python3
"""
Generate API Key for Papr Memory Open Source

This script generates a secure API key and stores it in Parse Server.
The API key can be used to authenticate requests to the Papr Memory API.

Usage:
    python scripts/generate_api_key.py --email developer@example.com --name "My Project"

    # With custom rate limit
    python scripts/generate_api_key.py --email dev@example.com --name "My Project" --rate-limit 5000

Requirements:
    - Parse Server must be running
    - PARSE_SERVER_URL, PARSE_APPLICATION_ID, and PARSE_MASTER_KEY must be set in environment
"""

import os
import sys
import secrets
import requests
import argparse
from datetime import datetime


def generate_api_key():
    """Generate a secure API key with prefix for easy identification"""
    return f"pmem_oss_{secrets.token_urlsafe(32)}"


def create_api_key(email: str, name: str, rate_limit: int = 1000, enabled: bool = True):
    """
    Create an API key in Parse Server

    Args:
        email: Email address associated with the API key
        name: Descriptive name for the API key (e.g., "My Project")
        rate_limit: Requests per hour limit (default: 1000)
        enabled: Whether the API key is active (default: True)

    Returns:
        The generated API key string, or None if creation failed
    """
    # Get Parse Server configuration from environment
    parse_url = os.getenv("PARSE_SERVER_URL", "http://localhost:1337/parse")
    app_id = os.getenv("PARSE_APPLICATION_ID") or os.getenv("PARSE_SERVER_APPLICATION_ID")
    master_key = os.getenv("PARSE_MASTER_KEY") or os.getenv("PARSE_SERVER_MASTER_KEY")

    if not app_id or not master_key:
        print("❌ Error: PARSE_APPLICATION_ID and PARSE_MASTER_KEY must be set")
        print("\nMake sure you have .env file configured with:")
        print("  PARSE_APPLICATION_ID=your-app-id")
        print("  PARSE_MASTER_KEY=your-master-key")
        sys.exit(1)

    # Generate secure API key
    api_key = generate_api_key()

    print(f"\n🔑 Generating API key for {email}...")
    print(f"📝 Name: {name}")
    print(f"⚡ Rate limit: {rate_limit} requests/hour")

    # Create API key in Parse Server
    headers = {
        "X-Parse-Application-Id": app_id,
        "X-Parse-Master-Key": master_key,
        "Content-Type": "application/json"
    }

    data = {
        "api_key": api_key,
        "email": email,
        "name": name,
        "created_at": {"__type": "Date", "iso": datetime.utcnow().isoformat() + "Z"},
        "rate_limit": rate_limit,
        "enabled": enabled,
        "usage_count": 0,
        "last_used": None
    }

    try:
        response = requests.post(
            f"{parse_url}/classes/APIKey",
            headers=headers,
            json=data,
            timeout=10
        )

        if response.status_code in [200, 201]:
            result = response.json()
            object_id = result.get("objectId")

            print(f"\n{'='*70}")
            print("✅ API Key Generated Successfully!")
            print(f"{'='*70}")
            print(f"\n🔑 API Key: {api_key}")
            print(f"📧 Email: {email}")
            print(f"📝 Name: {name}")
            print(f"🆔 Object ID: {object_id}")
            print(f"⚡ Rate Limit: {rate_limit} requests/hour")
            print(f"\n{'='*70}")
            print("⚠️  IMPORTANT: Save this API key now!")
            print("   You won't be able to see it again.")
            print(f"{'='*70}")
            print("\n📋 Add to your .env file:")
            print(f"   PAPR_API_KEY={api_key}")
            print("\n📝 Use in API requests:")
            print(f"   curl -H 'X-API-Key: {api_key}' \\")
            print(f"        -H 'Content-Type: application/json' \\")
            print(f"        -d '{{\"content\": \"Test memory\"}}' \\")
            print(f"        http://localhost:5001/v1/memory")
            print()

            return api_key
        else:
            print(f"\n❌ Failed to create API key")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error connecting to Parse Server: {e}")
        print(f"\nMake sure Parse Server is running at: {parse_url}")
        print("You can check with: curl http://localhost:1337/parse/health")
        return None


def list_api_keys():
    """List all API keys in Parse Server"""
    parse_url = os.getenv("PARSE_SERVER_URL", "http://localhost:1337/parse")
    app_id = os.getenv("PARSE_APPLICATION_ID") or os.getenv("PARSE_SERVER_APPLICATION_ID")
    master_key = os.getenv("PARSE_MASTER_KEY") or os.getenv("PARSE_SERVER_MASTER_KEY")

    headers = {
        "X-Parse-Application-Id": app_id,
        "X-Parse-Master-Key": master_key,
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(
            f"{parse_url}/classes/APIKey",
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            results = response.json().get("results", [])

            if not results:
                print("\n📭 No API keys found")
                return

            print(f"\n{'='*70}")
            print(f"📋 API Keys ({len(results)} total)")
            print(f"{'='*70}\n")

            for key in results:
                status = "✅ Enabled" if key.get("enabled", True) else "❌ Disabled"
                print(f"🔑 {key.get('name', 'Unnamed')}")
                print(f"   Email: {key.get('email', 'N/A')}")
                print(f"   Key: {key.get('api_key', 'N/A')[:20]}...")
                print(f"   Status: {status}")
                print(f"   Rate Limit: {key.get('rate_limit', 'N/A')} req/hour")
                print(f"   Usage: {key.get('usage_count', 0)} requests")
                print(f"   Created: {key.get('createdAt', 'N/A')}")
                print()
        else:
            print(f"❌ Failed to list API keys: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to Parse Server: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate API keys for Papr Memory Open Source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate API key with default settings
  python scripts/generate_api_key.py --email dev@example.com --name "My Project"

  # Generate API key with custom rate limit
  python scripts/generate_api_key.py --email dev@example.com --name "My Project" --rate-limit 5000

  # List all API keys
  python scripts/generate_api_key.py --list
        """
    )

    parser.add_argument("--email", help="Email address for the API key owner")
    parser.add_argument("--name", help="Descriptive name for the API key (e.g., 'My Project')")
    parser.add_argument("--rate-limit", type=int, default=1000,
                       help="Requests per hour limit (default: 1000)")
    parser.add_argument("--disabled", action="store_true",
                       help="Create API key in disabled state")
    parser.add_argument("--list", action="store_true",
                       help="List all existing API keys")

    args = parser.parse_args()

    # Load environment variables from .env file if it exists
    env_file = ".env.opensource" if os.path.exists(".env.opensource") else ".env"
    if os.path.exists(env_file):
        print(f"📁 Loading environment from {env_file}")
        from dotenv import load_dotenv
        load_dotenv(env_file)

    if args.list:
        list_api_keys()
        return

    if not args.email or not args.name:
        parser.print_help()
        print("\n❌ Error: --email and --name are required")
        sys.exit(1)

    create_api_key(
        email=args.email,
        name=args.name,
        rate_limit=args.rate_limit,
        enabled=not args.disabled
    )


if __name__ == "__main__":
    main()
