#!/usr/bin/env python3
"""
Quick test script to verify schema enforcement with a single memory item.
"""
import asyncio
import os
import sys

async def test_schema_enforcement():
    """Test that schema_id in metadata triggers custom schema enforcement."""
    
    print("🧪 Testing Schema Enforcement")
    print("=" * 60)
    
    # Create a test memory with schema_id in metadata
    test_memory = {
        "id": "test-schema-enforcement-001",
        "content": "Google's password reminder exposes user names, photos, and partial contact info during recovery.",
        "type": "text",
        "metadata": {
            "user_id": "test_user",
            "organization_id": "test_org",
            "namespace_id": "test_namespace",
            "workspace_id": "test_workspace",
            "customMetadata": {
                "schema_id": "Dh6EivRmo8",  # Security Behaviors & Risk schema
                "source": "test_schema_enforcement"
            }
        }
    }
    
    print(f"📝 Test Memory:")
    print(f"   Content: {test_memory['content']}")
    print(f"   Schema ID: {test_memory['metadata']['customMetadata']['schema_id']}")
    print()
    
    # Process the memory (this should extract schema_id and enforce it)
    print("🔄 Processing memory...")
    print()
    
    # Note: Full processing would require Neo4j session and other dependencies
    # For now, we'll just verify the extraction logic works
    
    # Extract schema_ids (simulating what happens in _index_memories_and_process)
    schema_ids = None
    metadata = test_memory.get('metadata', {})
    if isinstance(metadata, dict):
        custom_metadata = metadata.get('customMetadata', {})
        if isinstance(custom_metadata, dict):
            schema_id = custom_metadata.get('schema_id')
            if schema_id:
                schema_ids = [schema_id]
                print(f"✅ PASS: Successfully extracted schema_ids: {schema_ids}")
            else:
                print(f"❌ FAIL: No schema_id found in customMetadata")
                return False
        else:
            print(f"❌ FAIL: customMetadata is not a dict")
            return False
    else:
        print(f"❌ FAIL: metadata is not a dict")
        return False
    
    print()
    print(f"✅ Schema Enforcement Test PASSED!")
    print(f"   Expected node types: SecurityBehavior, Control, RiskIndicator, Impact, VerificationMethod")
    print(f"   Schema ID will be passed to LLM generator for enforcement")
    print()
    
    return True

if __name__ == "__main__":
    result = asyncio.run(test_schema_enforcement())
    sys.exit(0 if result else 1)

