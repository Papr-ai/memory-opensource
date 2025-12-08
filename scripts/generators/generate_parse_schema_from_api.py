#!/usr/bin/env python3
"""
Generate Parse Server Schema Initialization Script from Parse Server REST API

This script:
1. Connects to Parse Server REST API
2. Fetches all schemas via GET /schemas
3. Generates a Python script to initialize Parse Server for open-source deployment

This is SAFER than direct MongoDB access as it:
- Uses Parse Server's built-in schema API
- No risk of accidental writes
- Respects Parse Server permissions

Usage:
    python generate_parse_schema_from_api.py \
        --parse-url https://your-parse-server.com/parse \
        --app-id YOUR_APP_ID \
        --master-key YOUR_MASTER_KEY \
        --output scripts/init_parse_schema.py
"""

import requests
import argparse
import json
import sys
import os
import stat


def fetch_all_schemas(parse_url: str, app_id: str, master_key: str):
    """
    Fetch all Parse Server schemas via REST API and clean them for creation.

    Uses GET /schemas endpoint which is READ-ONLY.
    """
    headers = {
        "X-Parse-Application-Id": app_id,
        "X-Parse-Master-Key": master_key,
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(f"{parse_url}/schemas", headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        schemas = data.get("results", [])

        # System fields that Parse Server manages automatically
        system_fields_to_remove = {'ACL', 'objectId', 'createdAt', 'updatedAt'}

        # Azure CosmosDB/DocumentDB specific fields
        azure_fields_to_remove = {'DocumentDBDefaultIndex', '_id'}

        # Built-in class protected fields that can't be explicitly added
        builtin_protected_fields = {
            '_Installation': {'GCMSenderId', 'deviceType', 'installationId', 'deviceToken',
                             'badge', 'timeZone', 'localeIdentifier', 'pushType', 'channels'},
            '_Session': {'createdWith', 'restricted', 'user', 'installationId', 'sessionToken', 'expiresAt'},
            '_Role': {'name', 'users', 'roles'},
            '_User': {'username', 'password', 'email', 'emailVerified', 'authData'}
        }

        # Built-in Parse classes that are auto-created - skip these entirely
        builtin_classes = {'_User', '_Session', '_Role', '_Installation'}

        schemas_to_create = []

        for schema in schemas:
            class_name = schema.get('className')

            # Skip built-in Parse classes - they're auto-created
            if class_name in builtin_classes:
                continue

            # Remove indexes (Parse Server manages these separately)
            if 'indexes' in schema:
                del schema['indexes']

            # Clean up classLevelPermissions
            if 'classLevelPermissions' in schema:
                clp = schema['classLevelPermissions']
                # Remove 'ACL' from permissions as it's not a valid operation
                if 'ACL' in clp:
                    del clp['ACL']
                # Remove 'addField' if present (not needed for schema creation)
                if 'addField' in clp:
                    del clp['addField']

            # Filter fields
            if 'fields' in schema:
                fields_to_keep = {}
                protected_fields_for_class = builtin_protected_fields.get(class_name, set())

                for field_name, field_def in schema['fields'].items():
                    # Skip system fields
                    if field_name in system_fields_to_remove:
                        continue
                    # Skip Azure-specific fields
                    if field_name in azure_fields_to_remove:
                        continue
                    # Skip protected fields for built-in classes
                    if field_name in protected_fields_for_class:
                        continue

                    # Clean up field definition
                    if isinstance(field_def, dict):
                        # Remove index information from field definition
                        field_def_clean = {k: v for k, v in field_def.items()
                                         if k not in ['__type', 'indexes']}
                        fields_to_keep[field_name] = field_def_clean
                    else:
                        fields_to_keep[field_name] = field_def

                schema['fields'] = fields_to_keep

            schemas_to_create.append(schema)

        return schemas_to_create

    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching schemas from Parse Server: {e}")
        sys.exit(1)


def generate_init_script(schemas: list, output_file: str):
    """
    Generate a Python script that initializes Parse Server with the extracted schemas.
    """
    import pprint

    # Sort classes - put built-in classes first
    builtin_classes = ['_User', '_Role', '_Session', '_Installation']

    schemas_dict = {schema['className']: schema for schema in schemas}

    sorted_class_names = []
    for cls in builtin_classes:
        if cls in schemas_dict:
            sorted_class_names.append(cls)

    for cls in sorted(schemas_dict.keys()):
        if cls not in builtin_classes:
            sorted_class_names.append(cls)

    # Reorder schemas
    sorted_schemas = {cls: schemas_dict[cls] for cls in sorted_class_names if cls in schemas_dict}

    # Format schemas as Python code (not JSON) so booleans are True/False not true/false
    schemas_python_str = pprint.pformat(sorted_schemas, indent=4, width=120)

    script_content = f'''#!/usr/bin/env python3
"""
Initialize Parse Server Schema for Papr Memory Open Source

This script creates all required Parse classes with proper fields and permissions.
Auto-generated from development Parse Server REST API.

Usage:
    python {os.path.basename(output_file)} --parse-url http://localhost:1337/parse --app-id YOUR_APP_ID --master-key YOUR_MASTER_KEY
"""

import os
import sys
import requests
import json
import argparse
from typing import Dict, Any


def create_or_update_schema(parse_url: str, app_id: str, master_key: str, schema: Dict[str, Any]) -> bool:
    """
    Create or update a Parse Server schema via REST API.
    """
    class_name = schema["className"]

    headers = {{
        "X-Parse-Application-Id": app_id,
        "X-Parse-Master-Key": master_key,
        "Content-Type": "application/json"
    }}

    # Check if schema already exists
    check_url = f"{{parse_url}}/schemas/{{class_name}}"
    response = requests.get(check_url, headers=headers)

    if response.status_code == 200:
        # Schema exists, update it
        print(f"Schema '{{class_name}}' exists, updating...")
        response = requests.put(check_url, headers=headers, json=schema)
    else:
        # Schema doesn't exist, create it
        print(f"Creating schema '{{class_name}}'...")
        response = requests.post(f"{{parse_url}}/schemas", headers=headers, json=schema)

    if response.status_code in [200, 201]:
        print(f"✓ Schema '{{class_name}}' {{'updated' if response.status_code == 200 else 'created'}} successfully")
        return True
    else:
        print(f"✗ Failed to create/update schema '{{class_name}}': {{response.status_code}}")
        print(f"  Response: {{response.text}}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Initialize Parse Server schema")
    parser.add_argument("--parse-url", default=os.getenv("PARSE_SERVER_URL", "http://localhost:1337/parse"),
                       help="Parse Server URL (default: http://localhost:1337/parse)")
    parser.add_argument("--app-id", default=os.getenv("PARSE_APPLICATION_ID"),
                       help="Parse Application ID")
    parser.add_argument("--master-key", default=os.getenv("PARSE_MASTER_KEY"),
                       help="Parse Master Key")
    parser.add_argument("--skip-builtin", action="store_true",
                       help="Skip built-in Parse classes (_User, _Role, _Session)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be created without actually creating")

    args = parser.parse_args()

    if not args.app_id or not args.master_key:
        print("Error: --app-id and --master-key are required (or set PARSE_APPLICATION_ID and PARSE_MASTER_KEY)")
        sys.exit(1)

    print(f"Initializing Parse Server at {{args.parse_url}}")
    print(f"Application ID: {{args.app_id}}")
    if args.dry_run:
        print("DRY RUN MODE: No changes will be made")
    print()

    # Schemas extracted from development Parse Server
    # Using pprint to get Python-compatible dict (True/False instead of true/false)
    schemas = {schemas_python_str}

    success_count = 0
    failed_count = 0
    skipped_count = 0

    for class_name, schema in schemas.items():
        # Skip built-in classes if requested
        if args.skip_builtin and class_name.startswith('_'):
            print(f"Skipping built-in class '{{class_name}}'")
            skipped_count += 1
            continue

        if args.dry_run:
            field_count = len(schema.get('fields', {{}}))
            print(f"Would create/update schema '{{class_name}}' ({{field_count}} fields)")
            success_count += 1
            continue

        if create_or_update_schema(args.parse_url, args.app_id, args.master_key, schema):
            success_count += 1
        else:
            failed_count += 1

    print()
    print(f"Schema initialization complete!")
    print(f"  ✓ Successful: {{success_count}}")
    if skipped_count > 0:
        print(f"  ⊘ Skipped: {{skipped_count}}")
    if failed_count > 0:
        print(f"  ✗ Failed: {{failed_count}}")

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
'''

    with open(output_file, 'w') as f:
        f.write(script_content)

    # Make it executable
    os.chmod(output_file, os.stat(output_file).st_mode | stat.S_IEXEC)

    print(f"✓ Generated initialization script: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract Parse Server schema from REST API")
    parser.add_argument("--parse-url", required=True,
                       help="Parse Server URL (e.g., https://your-server.com/parse)")
    parser.add_argument("--app-id", required=True,
                       help="Parse Application ID")
    parser.add_argument("--master-key", required=True,
                       help="Parse Master Key")
    parser.add_argument("--output", default="scripts/init_parse_schema.py",
                       help="Output Python script path (default: scripts/init_parse_schema.py)")

    args = parser.parse_args()

    print(f"Fetching schemas from Parse Server: {args.parse_url}")
    print("This is a READ-ONLY operation - no data will be modified")
    print()

    # Fetch schemas via REST API (READ-ONLY)
    schemas = fetch_all_schemas(args.parse_url, args.app_id, args.master_key)

    if not schemas:
        print("✗ No schemas found")
        sys.exit(1)

    print(f"✓ Found {len(schemas)} Parse classes:")
    for schema in schemas:
        class_name = schema['className']
        field_count = len(schema.get('fields', {}))
        print(f"  - {class_name} ({field_count} fields)")

    # Generate initialization script
    print(f"\nGenerating initialization script: {args.output}")
    generate_init_script(schemas, args.output)

    print("\n✓ Done! You can now run:")
    print(f"  python {args.output} --parse-url http://localhost:1337/parse --app-id YOUR_APP_ID --master-key YOUR_MASTER_KEY")
    print("\nOr test with dry-run first:")
    print(f"  python {args.output} --parse-url http://localhost:1337/parse --app-id YOUR_APP_ID --master-key YOUR_MASTER_KEY --dry-run")


if __name__ == "__main__":
    main()
