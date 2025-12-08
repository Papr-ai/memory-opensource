#!/usr/bin/env python3
"""
Test script to verify MessageRole and category type enforcement
"""

from models.shared_types import MemoryMetadata, MessageRole, UserMemoryCategory, AssistantMemoryCategory
from pydantic import ValidationError

def test_category_types():
    """Test that category types are properly enforced"""
    
    print("🧪 Testing category type enforcement...")
    
    # Test 1: Valid user categories
    print("\n1. Testing valid user categories:")
    for category in UserMemoryCategory:
        try:
            metadata = MemoryMetadata(
                role=MessageRole.USER,
                category=category
            )
            print(f"   ✅ USER + {category.value}: {type(metadata.category)} = {metadata.category}")
        except ValidationError as e:
            print(f"   ❌ USER + {category.value}: {e}")
    
    # Test 2: Valid assistant categories
    print("\n2. Testing valid assistant categories:")
    for category in AssistantMemoryCategory:
        try:
            metadata = MemoryMetadata(
                role=MessageRole.ASSISTANT,
                category=category
            )
            print(f"   ✅ ASSISTANT + {category.value}: {type(metadata.category)} = {metadata.category}")
        except ValidationError as e:
            print(f"   ❌ ASSISTANT + {category.value}: {e}")
    
    # Test 3: String values should be converted to enums
    print("\n3. Testing string to enum conversion:")
    try:
        metadata = MemoryMetadata(
            role=MessageRole.USER,
            category="fact"  # String should be converted to UserMemoryCategory.FACT
        )
        print(f"   ✅ USER + 'fact' string: {type(metadata.category)} = {metadata.category}")
    except ValidationError as e:
        print(f"   ❌ USER + 'fact' string: {e}")
    
    # Test 4: Invalid category for role should fail
    print("\n4. Testing invalid category for role:")
    try:
        metadata = MemoryMetadata(
            role=MessageRole.USER,
            category="skills"  # Assistant category for user role
        )
        print(f"   ❌ Should have failed: USER + 'skills': {metadata.category}")
    except ValidationError as e:
        print(f"   ✅ Correctly failed: USER + 'skills': {e}")
    
    # Test 5: Invalid category string should fail
    print("\n5. Testing invalid category string:")
    try:
        metadata = MemoryMetadata(
            role=MessageRole.USER,
            category="invalid_category"
        )
        print(f"   ❌ Should have failed: USER + 'invalid_category': {metadata.category}")
    except ValidationError as e:
        print(f"   ✅ Correctly failed: USER + 'invalid_category': {e}")
    
    print("\n🎉 Category type enforcement test completed!")

if __name__ == "__main__":
    test_category_types()
