"""
Migration script to add BatchMemoryRequest class schema to Parse Server

This script adds all the required fields for the BatchMemoryRequest class
which is used to avoid Temporal gRPC payload limits for batch memory processing.

Usage:
    python scripts/migrate_batch_memory_request.py
"""

import os
import asyncio
import httpx
from dotenv import load_dotenv, find_dotenv

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)


async def migrate_batch_memory_request_schema():
    """Add BatchMemoryRequest class schema to Parse Server"""

    # Get Parse Server configuration
    parse_url = os.getenv("PARSE_SERVER_URL") or os.getenv("PARSE_SERVER_URL")
    app_id = os.getenv("PARSE_APPLICATION_ID")
    master_key = os.getenv("PARSE_MASTER_KEY")

    if not all([parse_url, app_id, master_key]):
        print("❌ Missing required environment variables:")
        print(f"   PARSE_SERVER_URL: {parse_url}")
        print(f"   PARSE_APPLICATION_ID: {app_id}")
        print(f"   PARSE_MASTER_KEY: {master_key}")
        return False

    print(f"🔧 Migrating BatchMemoryRequest schema to Parse Server: {parse_url}")

    # Define schema (do NOT include built-in fields like objectId/createdAt/updatedAt/ACL)
    schema = {
        "className": "BatchMemoryRequest",
        "fields": {
            # Identifiers
            "batchId": {"type": "String", "required": True},
            "requestId": {"type": "String"},

            # Multi-tenant context (Pointers)
            "organization": {
                "type": "Pointer",
                "targetClass": "Organization"
            },
            "namespace": {
                "type": "Pointer",
                "targetClass": "Namespace"
            },
            "user": {
                "type": "Pointer",
                "targetClass": "_User"
            },
            "workspace": {
                "type": "Pointer",
                "targetClass": "WorkSpace"
            },

            # Batch data storage
            "batchDataFile": {"type": "File"},
            "batchMetadata": {"type": "Object"},

            # Processing status
            "status": {
                "type": "String",
                "required": True,
                "defaultValue": "pending"
            },
            "processedCount": {
                "type": "Number",
                "defaultValue": 0
            },
            "successCount": {
                "type": "Number",
                "defaultValue": 0
            },
            "failCount": {
                "type": "Number",
                "defaultValue": 0
            },
            "totalMemories": {
                "type": "Number",
                "defaultValue": 0
            },

            # Temporal tracking
            "workflowId": {"type": "String"},
            "workflowRunId": {"type": "String"},

            # Webhook configuration
            "webhookUrl": {"type": "String"},
            "webhookSecret": {"type": "String"},
            "webhookSent": {
                "type": "Boolean",
                "defaultValue": False
            },

            # Timing metadata
            "startedAt": {"type": "Date"},
            "completedAt": {"type": "Date"},
            "processingDurationMs": {"type": "Number"},

            # Error tracking
            "errors": {"type": "Array"}
        },
        "classLevelPermissions": {
            "find": {"requiresAuthentication": True},
            "count": {"requiresAuthentication": True},
            "get": {"requiresAuthentication": True},
            "create": {"requiresAuthentication": True},
            "update": {"requiresAuthentication": True},
            "delete": {"requiresAuthentication": True},
            "addField": {},
            "protectedFields": {}
        },
        "indexes": {
            # Single field indexes
            "batchId_1": {"batchId": 1},
            "status_1": {"status": 1},
            "createdAt_1": {"createdAt": 1},
            "workflowId_1": {"workflowId": 1},

            # Compound indexes
            "organization_namespace": {"organization": 1, "namespace": 1},
            "user_status": {"user": 1, "status": 1},
            "status_createdAt": {"status": 1, "createdAt": 1}
        }
    }

    headers = {
        "X-Parse-Application-Id": app_id,
        "X-Parse-Master-Key": master_key,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Check if class already exists
        print("📋 Checking if BatchMemoryRequest class exists...")

        try:
            get_response = await client.get(
                f"{parse_url}/parse/schemas/BatchMemoryRequest",
                headers=headers
            )

            if get_response.status_code == 200:
                print("✅ BatchMemoryRequest class already exists")
                print("🔄 Updating schema...")

                # Update existing schema
                update_response = await client.put(
                    f"{parse_url}/parse/schemas/BatchMemoryRequest",
                    headers=headers,
                    json=schema
                )

                if update_response.status_code == 200:
                    print("✅ Schema updated successfully!")
                    print(f"   Response: {update_response.json()}")
                    return True
                else:
                    print(f"❌ Failed to update schema: {update_response.status_code}")
                    print(f"   Response: {update_response.text}")
                    return False

            else:
                # Class doesn't exist, create it
                print("📝 Creating new BatchMemoryRequest class...")

                create_response = await client.post(
                    f"{parse_url}/parse/schemas/BatchMemoryRequest",
                    headers=headers,
                    json=schema
                )

                if create_response.status_code in [200, 201]:
                    print("✅ BatchMemoryRequest class created successfully!")
                    print(f"   Response: {create_response.json()}")
                    return True
                else:
                    print(f"❌ Failed to create class: {create_response.status_code}")
                    print(f"   Response: {create_response.text}")
                    return False

        except Exception as e:
            print(f"❌ Error during migration: {e}")
            return False


async def verify_schema():
    """Verify the BatchMemoryRequest schema was created correctly"""

    parse_url = os.getenv("PARSE_SERVER_URL") or os.getenv("PARSE_SERVER_URL")
    app_id = os.getenv("PARSE_APPLICATION_ID")
    master_key = os.getenv("PARSE_MASTER_KEY")

    headers = {
        "X-Parse-Application-Id": app_id,
        "X-Parse-Master-Key": master_key,
        "Content-Type": "application/json"
    }

    print("\n🔍 Verifying schema...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{parse_url}/parse/schemas/BatchMemoryRequest",
            headers=headers
        )

        if response.status_code == 200:
            schema_data = response.json()
            fields = schema_data.get("fields", {})
            indexes = schema_data.get("indexes", {})

            print("✅ Schema verification successful!")
            print(f"\n📊 Fields ({len(fields)}):")
            for field_name, field_type in sorted(fields.items()):
                if isinstance(field_type, dict):
                    type_str = field_type.get("type", "Unknown")
                    target = field_type.get("targetClass", "")
                    if target:
                        type_str = f"{type_str} → {target}"
                else:
                    type_str = str(field_type)
                print(f"   • {field_name}: {type_str}")

            print(f"\n📇 Indexes ({len(indexes)}):")
            for index_name, index_def in sorted(indexes.items()):
                print(f"   • {index_name}: {index_def}")

            return True
        else:
            print(f"❌ Failed to verify schema: {response.status_code}")
            print(f"   Response: {response.text}")
            return False


async def main():
    """Main migration function"""
    print("=" * 70)
    print("   Parse Server BatchMemoryRequest Schema Migration")
    print("=" * 70)
    print()

    # Run migration
    success = await migrate_batch_memory_request_schema()

    if success:
        # Verify schema
        await verify_schema()

        print("\n" + "=" * 70)
        print("✅ Migration completed successfully!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. The BatchMemoryRequest class is now ready to use")
        print("2. You can test it with the batch processing workflow")
        print("3. Run the unit tests to verify functionality")
        print()
    else:
        print("\n" + "=" * 70)
        print("❌ Migration failed!")
        print("=" * 70)
        print()
        print("Please check the error messages above and:")
        print("1. Verify your Parse Server is running")
        print("2. Check your environment variables are correct")
        print("3. Ensure you have master key access")
        print()


if __name__ == "__main__":
    asyncio.run(main())
