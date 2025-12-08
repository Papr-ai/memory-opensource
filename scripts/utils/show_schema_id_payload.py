#!/usr/bin/env python3
"""
Show the schema_id memory payload format
"""

import json

# Configuration
SCHEMA_ID = "IeskhPibBx"  # From successful test
EXTERNAL_USER_ID = "security_test_user_004"

def get_memory_data() -> dict:
    """Get the memory data for schema_id approach"""
    return {
        "type": "text",
        "content": "Security incident detected: SQL injection attempt targeting /api/users endpoint from IP 192.168.1.100. This is a credential access tactic with high severity impact on data confidentiality.",
        "schema_id": SCHEMA_ID,
        "metadata": {
            "event_type": "security_incident",
            "test_type": "schema_id_approach_only",
            "external_user_id": EXTERNAL_USER_ID
        }
    }

def main():
    print("🚀 Schema ID Memory Payload")
    print("=" * 60)
    print(f"Schema ID: {SCHEMA_ID}")
    print(f"External User ID: {EXTERNAL_USER_ID}")
    print("=" * 60)
    
    memory_data = get_memory_data()
    
    print("\n📝 Memory Payload (JSON):")
    print(json.dumps(memory_data, indent=2))
    
    print(f"\n🌐 API Call would be:")
    print(f"POST http://localhost:8000/v1/memory?external_user_id={EXTERNAL_USER_ID}")
    print(f"Authorization: Bearer <API_KEY>")
    print(f"Content-Type: application/json")
    
    print(f"\n✅ The schema_id approach payload is correctly formatted!")
    print(f"   - Uses existing schema: {SCHEMA_ID}")
    print(f"   - Proper type: 'text'")
    print(f"   - External user ID in query param: {EXTERNAL_USER_ID}")
    print(f"   - Metadata includes test type and event type")

if __name__ == "__main__":
    main()

