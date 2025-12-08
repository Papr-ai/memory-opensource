#!/usr/bin/env python3
"""
Test script for enum validation in PropertyDefinition.

This script tests the enum_values validation logic to ensure it works correctly.
"""

from models.user_schemas import PropertyDefinition, PropertyType
from pydantic import ValidationError
import json

def test_valid_enums():
    """Test valid enum configurations"""
    print("✅ Testing valid enum configurations...")
    
    # Valid enum with 5 values
    prop1 = PropertyDefinition(
        type=PropertyType.STRING,
        enum_values=["small", "medium", "large", "xl", "xxl"],
        description="Size options"
    )
    print(f"   ✓ Valid enum with 5 values: {prop1.enum_values}")
    
    # Valid enum with default value
    prop2 = PropertyDefinition(
        type=PropertyType.STRING,
        enum_values=["bronze", "silver", "gold", "platinum"],
        default="bronze",
        description="Membership tier"
    )
    print(f"   ✓ Valid enum with default: {prop2.enum_values}, default: {prop2.default}")
    
    # Maximum 10 values
    prop3 = PropertyDefinition(
        type=PropertyType.STRING,
        enum_values=[f"option_{i}" for i in range(1, 11)],  # 10 values
        description="Max enum values"
    )
    print(f"   ✓ Valid enum with 10 values: {len(prop3.enum_values)} values")

def test_invalid_enums():
    """Test invalid enum configurations"""
    print("\n❌ Testing invalid enum configurations...")
    
    # Test too many values (>10)
    try:
        PropertyDefinition(
            type=PropertyType.STRING,
            enum_values=[f"option_{i}" for i in range(1, 12)],  # 11 values
            description="Too many enum values"
        )
        print("   ❌ FAILED: Should have rejected >10 enum values")
    except ValidationError as e:
        print("   ✓ Correctly rejected >10 enum values")
    
    # Test empty enum list
    try:
        PropertyDefinition(
            type=PropertyType.STRING,
            enum_values=[],
            description="Empty enum values"
        )
        print("   ❌ FAILED: Should have rejected empty enum list")
    except ValidationError as e:
        print("   ✓ Correctly rejected empty enum list")
    
    # Test duplicate values
    try:
        PropertyDefinition(
            type=PropertyType.STRING,
            enum_values=["small", "medium", "large", "medium"],
            description="Duplicate enum values"
        )
        print("   ❌ FAILED: Should have rejected duplicate enum values")
    except ValidationError as e:
        print("   ✓ Correctly rejected duplicate enum values")
    
    # Test empty string values
    try:
        PropertyDefinition(
            type=PropertyType.STRING,
            enum_values=["small", "", "large"],
            description="Empty string in enum values"
        )
        print("   ❌ FAILED: Should have rejected empty string in enum values")
    except ValidationError as e:
        print("   ✓ Correctly rejected empty string in enum values")
    
    # Test invalid default value
    try:
        PropertyDefinition(
            type=PropertyType.STRING,
            enum_values=["small", "medium", "large"],
            default="invalid_size",
            description="Invalid default value"
        )
        print("   ❌ FAILED: Should have rejected invalid default value")
    except ValidationError as e:
        print("   ✓ Correctly rejected invalid default value")

def test_enum_serialization():
    """Test that enums serialize correctly to JSON"""
    print("\n📄 Testing enum serialization...")
    
    prop = PropertyDefinition(
        type=PropertyType.STRING,
        enum_values=["electronics", "clothing", "books", "home", "sports"],
        default="electronics",
        required=True,
        description="Product category"
    )
    
    # Test model_dump (Pydantic v2)
    data = prop.model_dump()
    print(f"   ✓ Serialized to dict: {json.dumps(data, indent=2)}")
    
    # Test JSON serialization
    json_str = prop.model_dump_json()
    print(f"   ✓ Serialized to JSON: {json_str}")

def main():
    """Main test function"""
    print("🧪 Testing Enum Validation in PropertyDefinition")
    print("=" * 50)
    
    test_valid_enums()
    test_invalid_enums()
    test_enum_serialization()
    
    print("\n🏁 Enum validation tests completed!")

if __name__ == "__main__":
    main()

