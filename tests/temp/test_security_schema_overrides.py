#!/usr/bin/env python3

import os
import json
import httpx
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_security_schema_property_overrides():
    """Test property overrides with security schema node types"""
    
    logger.info("🚀 Testing Security Schema Property Overrides")
    
    # Security incident content that should trigger security schema selection
    memory_content = """
    SECURITY INCIDENT REPORT: Phishing attack detected targeting our development team.
    Alice (Security Analyst): We've identified a sophisticated phishing campaign targeting our developers.
    Bob (CISO): This appears to be a credential harvesting attempt. We need immediate containment.
    Charlie (IT Admin): I've already started rotating compromised credentials and blocking suspicious IPs.
    Alice: The attack vector was a fake GitHub security notification. We should implement additional MFA controls.
    """
    
    logger.info(f"📝 Security incident content: {memory_content[:100]}...")
    
    # Property overrides targeting SECURITY SCHEMA node types
    property_overrides = [
        # Match specific security behaviors by name and set their IDs
        {"nodeLabel": "SecurityBehavior", "match": {"name": "Phishing Detection"}, "set": {"id": "behavior_phishing_123", "priority": "critical"}},
        {"nodeLabel": "SecurityBehavior", "match": {"name": "Credential Rotation"}, "set": {"id": "behavior_cred_456", "automation": "enabled"}},
        {"nodeLabel": "SecurityBehavior", "match": {"name": "MFA Implementation"}, "set": {"id": "behavior_mfa_789", "scope": "development_team"}},
        
        # Apply to all Tactic nodes (no match condition)
        {"nodeLabel": "Tactic", "set": {"framework": "MITRE_ATT&CK", "category": "OVERRIDE_DEFENSE"}},
        
        # Apply to all Impact nodes
        {"nodeLabel": "Impact", "set": {"severity": "OVERRIDE_HIGH", "business_unit": "override_engineering"}}
    ]
    
    logger.info(f"🔧 Security schema property overrides: {json.dumps(property_overrides, indent=2)}")
    
    # Create memory with GraphGeneration - use auto mode to let LLM select schema
    test_id = f"security_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    memory_data = {
        "content": memory_content,
        "type": "text",
        "metadata": {
            "source": "security_incident",
            "timestamp": datetime.now().isoformat(),
            "event_type": "phishing_attack",
            "test_type": f"security_overrides_{test_id}"
        },
        "graph_generation": {
            "mode": "auto",
            "auto": {
                "schema_id": "IeskhPibBx",  # Force security schema selection
                "property_overrides": property_overrides
            }
        }
    }
    
    # API configuration
    BASE_URL = "http://localhost:8000"
    TEST_API_KEY = os.getenv("PAPR_API_KEY", "f80c5a2940f21882420b41690522cb2c")
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": TEST_API_KEY
    }
    
    # Make the request
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{BASE_URL}/v1/memory",
                params={"external_user_id": test_id},
                headers=headers,
                json=memory_data
            )
        
        if response.status_code == 200:
            logger.info("✅ Security schema memory created successfully!")
            logger.info(f"📄 Response: {json.dumps(response.json(), indent=2)}")
            
            logger.info(f"\n🔍 Checking logs for test_id: {test_id}")
            logger.info("Look for:")
            logger.info("  - Schema selection (should pick security schema)")
            logger.info("  - 🔧 MATCH CHECK logs showing security node matching")
            logger.info("  - 🔧 APPLIED OVERRIDES for SecurityBehavior, Tactic, Impact nodes")
            
        else:
            logger.error(f"❌ Request failed with status {response.status_code}")
            logger.error(f"Response: {response.text}")
            
    except Exception as e:
        logger.error(f"❌ Error making request: {e}")

if __name__ == "__main__":
    test_security_schema_property_overrides()
