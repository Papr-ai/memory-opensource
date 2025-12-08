#!/usr/bin/env python3
"""
Test script for custom schema index creation functionality.

This script tests the automatic Neo4j index creation for custom schemas
when they are registered as ACTIVE.
"""

import asyncio
import httpx
import json
import os
from datetime import datetime, timezone

# Test configuration
TEST_API_KEY = os.getenv("PAPR_API_KEY", "YQnxXIZPT0M9JVH3L0S0MNLicDaqJ4Vd")
BASE_URL = "http://localhost:8000"

# Sample custom schema with different property types
SAMPLE_SCHEMA = {
    "name": "Test E-commerce Schema",
    "description": "Test schema for index creation with various property types",
    "version": "1.0.0",
    "status": "active",  # This should trigger index creation
    "scope": "workspace",
    "node_types": {
        "Product": {
            "name": "Product",
            "label": "Product",
            "description": "E-commerce product",
            "properties": {
                "name": {
                    "type": "string",
                    "required": True,
                    "description": "Product name, typically 2-4 words describing the item (e.g., 'iPhone 15 Pro', 'Nike Air Max')"
                },
                "price": {
                    "type": "float", 
                    "required": True,
                    "description": "Product price in USD as decimal number (e.g., 999.99, 29.95)"
                },
                "sku": {
                    "type": "string",
                    "required": True,
                    "description": "Stock keeping unit - unique alphanumeric identifier (e.g., 'SKU-12345', 'PROD-ABC-001')"
                },
                "in_stock": {
                    "type": "boolean",
                    "required": True,
                    "description": "Availability status - true if currently available for purchase, false if out of stock"
                },
                "created_at": {
                    "type": "datetime",
                    "required": True,
                    "description": "Product creation timestamp in ISO 8601 format (e.g., '2024-01-15T10:30:00Z')"
                },
                "category": {
                    "type": "string",
                    "required": True,
                    "description": "Main product category - choose the most appropriate category for this item",
                    "enum_values": ["electronics", "clothing", "books", "home", "sports"]
                },
                "condition": {
                    "type": "string",
                    "required": False,
                    "description": "Physical condition - use 'new' for brand new items, 'like_new' for barely used, 'good' for minor wear",
                    "enum_values": ["new", "like_new", "good", "fair", "poor"],
                    "default": "new"
                },
                "description": {
                    "type": "string",
                    "required": False,
                    "description": "Product description"
                }
            },
            "required_properties": ["name", "price", "sku", "in_stock", "created_at", "category"],
            "unique_identifiers": ["sku"],
            "color": "#e74c3c",
            "icon": "product"
        },
        "Customer": {
            "name": "Customer",
            "label": "Customer", 
            "description": "E-commerce customer",
            "properties": {
                "email": {
                    "type": "string",
                    "required": True,
                    "description": "Customer email address in standard format (e.g., 'john.doe@example.com')"
                },
                "name": {
                    "type": "string",
                    "required": True,
                    "description": "Customer full name, first and last name (e.g., 'John Smith', 'Sarah Johnson')"
                },
                "age": {
                    "type": "integer",
                    "required": False,
                    "description": "Customer age in years as whole number (e.g., 25, 34, 67)"
                },
                "is_premium": {
                    "type": "boolean",
                    "required": True,
                    "description": "Premium membership status - true if customer has premium benefits, false for standard membership"
                },
                "membership_tier": {
                    "type": "string",
                    "required": True,
                    "description": "Membership level - 'bronze' for new customers, 'silver' for regular, 'gold' for loyal, 'platinum' for VIP",
                    "enum_values": ["bronze", "silver", "gold", "platinum"],
                    "default": "bronze"
                },
                "preferred_contact": {
                    "type": "string",
                    "required": False,
                    "description": "How customer prefers to be contacted - choose based on customer preference or default to email",
                    "enum_values": ["email", "phone", "sms", "mail"]
                }
            },
            "required_properties": ["email", "name", "is_premium", "membership_tier"],
            "unique_identifiers": ["email"],
            "color": "#3498db",
            "icon": "customer"
        }
    },
    "relationship_types": {
        "PURCHASED": {
            "name": "PURCHASED",
            "allowed_source_types": ["Customer"],
            "allowed_target_types": ["Product"],
            "description": "Customer purchased product"
        }
    }
}

async def test_schema_creation_and_indexes():
    """Test creating a custom schema and verify indexes are created"""
    
    print("🧪 Testing Custom Schema Index Creation")
    print("=" * 50)
    
    try:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
            headers = {
                'Content-Type': 'application/json',
                'X-API-Key': TEST_API_KEY,
                'X-Client-Type': 'test_script'
            }
            
            print("📝 Step 1: Creating custom schema...")
            
            # Create the schema
            response = await client.post('/v1/schemas', json=SAMPLE_SCHEMA, headers=headers)
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 201:
                result = response.json()
                if result.get('success'):
                    schema_data = result.get('data')
                    schema_id = schema_data.get('id')
                    schema_name = schema_data.get('name')
                    
                    print(f"✅ Schema created successfully!")
                    print(f"   Schema ID: {schema_id}")
                    print(f"   Schema Name: {schema_name}")
                    print(f"   Status: {schema_data.get('status')}")
                    print(f"   Node Types: {len(schema_data.get('node_types', {}))}")
                    
                    # Wait a moment for background index creation
                    print("\n⏳ Waiting for background index creation...")
                    await asyncio.sleep(3)
                    
                    print("\n🔍 Expected indexes that should be created:")
                    print("   For Product node:")
                    print("     - custom_product_name_idx (STRING)")
                    print("     - custom_product_price_idx (FLOAT)")
                    print("     - custom_product_sku_idx (STRING)")
                    print("     - custom_product_in_stock_idx (BOOLEAN)")
                    print("     - custom_product_created_at_idx (DATETIME)")
                    print("     - custom_product_category_idx (STRING with ENUM: electronics, clothing, books, home, sports)")
                    print("     - ACL indexes (user_id, workspace_id, etc.)")
                    print("   For Customer node:")
                    print("     - custom_customer_email_idx (STRING)")
                    print("     - custom_customer_name_idx (STRING)")
                    print("     - custom_customer_is_premium_idx (BOOLEAN)")
                    print("     - custom_customer_membership_tier_idx (STRING with ENUM: bronze, silver, gold, platinum)")
                    print("     - ACL indexes (user_id, workspace_id, etc.)")
                    
                    print(f"\n✅ Test completed! Check server logs for index creation details.")
                    print(f"   Look for logs containing: 'Creating indexes for custom schema {schema_id}'")
                    
                    return schema_id
                    
                else:
                    print(f"❌ Schema creation failed: {result.get('error')}")
                    return None
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return None

async def cleanup_test_schema(schema_id: str):
    """Clean up test schema (if deletion endpoint exists)"""
    if not schema_id:
        return
        
    print(f"\n🧹 Cleaning up test schema {schema_id}...")
    # Note: Add cleanup logic here if schema deletion endpoint is implemented
    print("   (Manual cleanup may be required)")

async def main():
    """Main test function"""
    print("🚀 Starting Custom Schema Index Creation Test")
    print(f"   Base URL: {BASE_URL}")
    print(f"   API Key: {TEST_API_KEY[:10]}...")
    print()
    
    schema_id = await test_schema_creation_and_indexes()
    
    if schema_id:
        await cleanup_test_schema(schema_id)
    
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    asyncio.run(main())

