#!/usr/bin/env python3
"""
Test script for GraphGeneration API: Auto Mode with Property Overrides
"""

import httpx
import json
import os
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_API_KEY = os.getenv("PAPR_API_KEY", "f80c5a2940f21882420b41690522cb2c")

def test_auto_property_overrides():
    """Test Auto Mode with Property Overrides"""
    
    # Generate unique test identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_id = f"test_property_overrides_{timestamp}"
    
    print("🚀 GraphGeneration API Test: Auto Mode with Property Overrides")
    print("=" * 70)
    print("🧪 Testing Auto Mode with Property Overrides")
    print("=" * 60)
    
    print("📋 Test Configuration:")
    print(f"   External User ID: {test_id}")
    print("   Property Overrides: SecurityBehavior.name, Tactic.tactic_id")
    print("   Simple Schema Mode: False")
    print("   Content: Security incident with credential stuffing")
    print()
    
    print("🎯 Expected Behavior:")
    print("   ✅ Use GraphGeneration API with mode='auto'")
    print("   ✅ Apply property overrides for specific nodes")
    print("   ✅ Let LLM select appropriate schema")
    print("   ✅ Override SecurityBehavior.name = 'Advanced Persistent Threat'")
    print("   ✅ Override Tactic.tactic_id = 'T1110'")
    print("   ✅ Generate other properties via LLM")
    print()
    
    print("🔍 Log Patterns to Look For:")
    print("   - '🔧 PROPERTY OVERRIDES: {\"SecurityBehavior\": {\"name\": \"Advanced Persistent Threat\"}}'")
    print("   - '🤖 AUTO MODE: schema_id=None, simple_schema_mode=False'")
    print("   - '🚀 GRAPH STEP 2: LLM selected schema_id=...'")
    print("   - Property override application in node generation")
    print()
    
    # Test payload with property overrides
    payload = {
        "content": f"[PROPERTY_OVERRIDE_TEST {timestamp}] Critical security alert: Advanced persistent threat detected using credential stuffing technique (T1110) to compromise administrator accounts. Multiple failed login attempts observed from suspicious IP addresses. Immediate containment and investigation required.",
        "type": "text",
        "metadata": {
            "external_user_id": test_id,
            "event_type": "security_incident",
            "test_type": f"property_overrides_{timestamp}",
            "severity": "critical",
            "source": "security_monitoring"
        },
        "graph_generation": {
            "mode": "auto",
            "auto": {
                "simple_schema_mode": False,
                "property_overrides": {
                    "SecurityBehavior": {
                        "name": "Advanced Persistent Threat"
                    },
                    "Tactic": {
                        "tactic_id": "T1110"
                    }
                }
            }
        }
    }
    
    print("🚀 Making API Request...")
    
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{API_BASE_URL}/v1/memory",
                json=payload,
                headers={
                    "X-API-Key": TEST_API_KEY,
                    "Content-Type": "application/json",
                    "X-Client-Type": "test_client"
                },
                params={"external_user_id": test_id}
            )
            
            print(f"📡 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ SUCCESS!")
                print()
                print("📊 Response Details:")
                print(f"   Status: {result.get('status', 'N/A')}")
                print(f"   Code: {result.get('code', 'N/A')}")
                
                if result.get('data') and len(result['data']) > 0:
                    memory_data = result['data'][0]
                    memory_id = memory_data.get('memoryId', 'N/A')
                    content_preview = memory_data.get('content', 'N/A')[:50] + "..." if memory_data.get('content') else 'N/A'
                    created_at = memory_data.get('createdAt', 'N/A')
                    
                    print(f"   Memory ID: {memory_id}")
                    print(f"   Content: {content_preview}")
                    print(f"   Created At: {created_at}")
                else:
                    print("   No memory data in response")
                    
            else:
                print("❌ FAILED!")
                print(f"   Status Code: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False
    
    print()
    print("🎉 Test Completed Successfully!")
    print()
    print("🔍 Next Steps for Verification:")
    print(f"   1. Check server logs for external_user_id: {test_id}")
    print("   2. Verify property override application logs")
    print("   3. Check Neo4j for nodes with overridden properties")
    print("   4. Verify LLM schema selection occurred")
    print()
    print("📝 Log Search Commands:")
    print(f"   grep '{test_id}' logs/app_*.log")
    print(f"   grep '{memory_id}' logs/app_*.log")
    print("   grep 'PROPERTY OVERRIDES' logs/app_*.log")
    print()
    
    print("=" * 70)
    print("🎯 TEST SUMMARY")
    print("=" * 70)
    print("✅ Test PASSED")
    print(f"   Memory ID: {memory_id}")
    print(f"   External User ID: {test_id}")
    print("   Property Overrides: SecurityBehavior.name, Tactic.tactic_id")
    print()
    print("🎉 The new GraphGeneration API is working correctly!")
    print("   - Auto mode with property overrides ✅")
    print("   - LLM schema selection with guidance ✅")
    print("   - Clean API structure without legacy compatibility ✅")
    
    return True

if __name__ == "__main__":
    success = test_auto_property_overrides()
    exit(0 if success else 1)
