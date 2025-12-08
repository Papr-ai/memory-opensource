#!/usr/bin/env python3
"""
Generate Parse Server Schema Initialization Script from Atlas MongoDB

This script:
1. Connects to Atlas MongoDB development database
2. Reads the _SCHEMA collection to extract Parse Server schema
3. Generates a Python script to initialize Parse Server for open-source deployment

Usage:
    python generate_parse_schema_from_atlas.py --atlas-uri "mongodb+srv://..." --output init_parse_schema.py
"""

import pymongo
import argparse
import json
from typing import Dict, List, Any
from collections import defaultdict


def parse_field_type(field_value: Any) -> Dict[str, Any]:
    """
    Parse a Parse Server field type definition.

    Parse Server stores types as strings like:
    - "String", "Number", "Boolean", "Date", "Object", "Array"
    - "*ClassName" for pointers (e.g., "*Organization", "*User")
    - "relation<ClassName>" for relations
    - "file" for files

    Returns a dict with 'type' and optional 'targetClass' for pointers/relations
    """
    if not isinstance(field_value, str):
        return {"type": "Object"}  # Default for non-string types

    field_type = field_value.strip()

    # Pointer type (*ClassName)
    if field_type.startswith('*'):
        target_class = field_type[1:]  # Remove the *
        return {
            "type": "Pointer",
            "targetClass": target_class
        }

    # Relation type (relation<ClassName>)
    if field_type.startswith('relation<') and field_type.endswith('>'):
        target_class = field_type[9:-1]  # Extract class name between < and >
        return {
            "type": "Relation",
            "targetClass": target_class
        }

    # File type
    if field_type.lower() == 'file':
        return {"type": "File"}

    # Standard types (String, Number, Boolean, Date, Object, Array)
    # Capitalize first letter for consistency with Parse Server API
    return {"type": field_type.capitalize()}


def get_parse_schemas(db) -> Dict[str, Dict]:
    """
    Extract all Parse Server schemas from the _SCHEMA collection.

    Returns a dict mapping class names to their schema definitions.
    """
    schemas = {}

    try:
        schema_collection = db['_SCHEMA']

        for schema_doc in schema_collection.find():
            class_name = schema_doc.get('_id')
            if not class_name:
                continue

            # Skip system fields that shouldn't be in the API schema
            reserved_fields = {'_id', 'objectId', 'createdAt', 'updatedAt', '_metadata', 'className'}

            # Extract fields
            fields = {}
            for field_name, field_value in schema_doc.items():
                if field_name in reserved_fields:
                    continue

                # Parse the field type
                field_def = parse_field_type(field_value)
                fields[field_name] = field_def

            # Extract CLPs (Class Level Permissions) from _metadata
            clps = {
                "find": {"*": True},
                "get": {"*": True},
                "create": {"requiresAuthentication": True},
                "update": {"requiresAuthentication": True},
                "delete": {"requiresAuthentication": True},
                "addField": {},
                "protectedFields": {}
            }

            if '_metadata' in schema_doc:
                metadata = schema_doc['_metadata']
                if 'class_permissions' in metadata:
                    # Parse Server stores permissions in a specific format
                    stored_perms = metadata['class_permissions']

                    # Map stored permissions to API format
                    for perm_type in ['find', 'get', 'create', 'update', 'delete', 'addField']:
                        if perm_type in stored_perms:
                            perm_value = stored_perms[perm_type]
                            if isinstance(perm_value, dict):
                                clps[perm_type] = perm_value
                            elif isinstance(perm_value, list):
                                # Convert list to dict format
                                clps[perm_type] = {role: True for role in perm_value}

                    # Protected fields
                    if 'protectedFields' in stored_perms:
                        clps['protectedFields'] = stored_perms['protectedFields']

            schemas[class_name] = {
                "className": class_name,
                "fields": fields,
                "classLevelPermissions": clps
            }

        return schemas

    except Exception as e:
        print(f"Error reading _SCHEMA collection: {e}")
        return {}


def generate_init_script(schemas: Dict[str, Dict], output_file: str):
    """
    Generate a Python script that initializes Parse Server with the extracted schemas.
    """

    # Sort classes - put built-in classes first
    builtin_classes = ['_User', '_Role', '_Session', '_Installation']
    custom_classes = [cls for cls in schemas.keys() if cls not in builtin_classes]
    sorted_classes = [cls for cls in builtin_classes if cls in schemas] + sorted(custom_classes)

    script_content = f'''#!/usr/bin/env python3
"""
Initialize Parse Server Schema for Papr Memory Open Source

This script creates all required Parse classes with proper fields and permissions.
Auto-generated from Atlas MongoDB development database.

Usage:
    python {output_file} --parse-url http://localhost:1337/parse --app-id YOUR_APP_ID --master-key YOUR_MASTER_KEY
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
        print(f"✓ Schema '{{class_name}}' {'updated' if response.status_code == 200 else 'created'} successfully")
        return True
    else:
        print(f"✗ Failed to create/update schema '{{class_name}}': {{response.status_code}}")
        print(f"  Response: {{response.text}}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Initialize Parse Server schema from Atlas MongoDB export")
    parser.add_argument("--parse-url", default=os.getenv("PARSE_SERVER_URL", "http://localhost:1337/parse"),
                       help="Parse Server URL (default: http://localhost:1337/parse)")
    parser.add_argument("--app-id", default=os.getenv("PARSE_APPLICATION_ID"),
                       help="Parse Application ID")
    parser.add_argument("--master-key", default=os.getenv("PARSE_MASTER_KEY"),
                       help="Parse Master Key")
    parser.add_argument("--skip-builtin", action="store_true",
                       help="Skip built-in Parse classes (_User, _Role, _Session)")

    args = parser.parse_args()

    if not args.app_id or not args.master_key:
        print("Error: --app-id and --master-key are required (or set PARSE_APPLICATION_ID and PARSE_MASTER_KEY)")
        sys.exit(1)

    print(f"Initializing Parse Server at {{args.parse_url}}")
    print(f"Application ID: {{args.app_id}}")
    print()

    # Schemas extracted from Atlas MongoDB
    schemas = {json.dumps(schemas, indent=4)}

    success_count = 0
    failed_count = 0

    for class_name, schema in schemas.items():
        # Skip built-in classes if requested
        if args.skip_builtin and class_name.startswith('_'):
            print(f"Skipping built-in class '{{class_name}}'")
            continue

        if create_or_update_schema(args.parse_url, args.app_id, args.master_key, schema):
            success_count += 1
        else:
            failed_count += 1

    print()
    print(f"Schema initialization complete!")
    print(f"  ✓ Successful: {{success_count}}")
    print(f"  ✗ Failed: {{failed_count}}")

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
'''

    with open(output_file, 'w') as f:
        f.write(script_content)

    # Make it executable
    import stat
    os.chmod(output_file, os.stat(output_file).st_mode | stat.S_IEXEC)

    print(f"✓ Generated initialization script: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract Parse Server schema from Atlas MongoDB")
    parser.add_argument("--atlas-uri", required=True,
                       help="MongoDB Atlas connection string (e.g., mongodb+srv://...)")
    parser.add_argument("--database", default="parsedev",
                       help="Database name (default: parsedev)")
    parser.add_argument("--output", default="scripts/init_parse_schema.py",
                       help="Output Python script path (default: scripts/init_parse_schema.py)")

    args = parser.parse_args()

    print(f"Connecting to Atlas MongoDB: {args.atlas_uri[:50]}...")

    try:
        client = pymongo.MongoClient(args.atlas_uri, serverSelectionTimeoutMS=10000)
        db = client[args.database]

        # Test connection
        client.server_info()
        print(f"✓ Connected to database: {args.database}")

        # Extract schemas
        print("\nExtracting Parse Server schemas from _SCHEMA collection...")
        schemas = get_parse_schemas(db)

        if not schemas:
            print("✗ No schemas found in _SCHEMA collection")
            sys.exit(1)

        print(f"✓ Found {len(schemas)} Parse classes:")
        for class_name in sorted(schemas.keys()):
            field_count = len(schemas[class_name]['fields'])
            print(f"  - {class_name} ({field_count} fields)")

        # Generate initialization script
        print(f"\nGenerating initialization script: {args.output}")
        generate_init_script(schemas, args.output)

        print("\n✓ Done! You can now run:")
        print(f"  python {args.output} --parse-url http://localhost:1337/parse --app-id YOUR_APP_ID --master-key YOUR_MASTER_KEY")

        client.close()

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
