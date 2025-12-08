#!/usr/bin/env python3

import asyncio
import json
import httpx
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
import os
API_BASE_URL = "http://localhost:8000"
TEST_API_KEY = os.getenv("PAPR_API_KEY", "f80c5a2940f21882420b41690522cb2c")

async def test_enhanced_property_overrides():
    """Test the new enhanced property_overrides with match conditions"""
    
    # Test data: conversation with 3 users
    memory_content = """
    Alice: Hey everyone, let's discuss the new project timeline.
    Bob: I think we should extend the deadline by 2 weeks.
    Charlie: I agree with Bob, but we need to consider the budget constraints.
    Alice: Good point Charlie. Let me check with the finance team.
    """
    
    # Enhanced property overrides with match conditions - using SYSTEM SCHEMA node types
    property_overrides = [
        # Match specific people by name and set their IDs (using "Person" not "User")
        {"nodeLabel": "Person", "match": {"name": "Alice"}, "set": {"id": "person_alice_123", "role": "project_manager"}},
        {"nodeLabel": "Person", "match": {"name": "Bob"}, "set": {"id": "person_bob_456", "department": "engineering"}},
        {"nodeLabel": "Person", "match": {"name": "Charlie"}, "set": {"id": "person_charlie_789", "department": "finance"}},
        
        # Apply to all Meeting nodes (no match condition)
        {"nodeLabel": "Meeting", "set": {"meeting_id": "meeting_override_2025_10_30", "type": "project_discussion_override"}},
        
        # Apply to all Task nodes
        {"nodeLabel": "Task", "set": {"priority": "OVERRIDE_HIGH", "department": "override_dept"}}
    ]
    
    # Create memory with GraphGeneration - use auto mode to let LLM select schema
    test_id = f"enhanced_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    memory_data = {
        "content": memory_content,
        "type": "text",
        "metadata": {
            "external_user_id": test_id,
            "source": "team_chat",
            "timestamp": datetime.now().isoformat(),
            "event_type": "team_conversation",
            "test_type": f"enhanced_overrides_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        },
        "graph_generation": {
            "mode": "auto",
            "auto": {
                "simple_schema_mode": False,
                "property_overrides": property_overrides
            }
        }
    }
    
    logger.info("🚀 Testing Enhanced Property Overrides")
    logger.info(f"📝 Memory content: {memory_content[:100]}...")
    logger.info(f"🔧 Property overrides: {json.dumps(property_overrides, indent=2)}")
    
    try:
        # Send request using the same method as the working test
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{API_BASE_URL}/v1/memory",
                json=memory_data,
                headers={
                    "X-API-Key": TEST_API_KEY,
                    "Content-Type": "application/json",
                    "X-Client-Type": "test_client"
                },
                params={"external_user_id": test_id}
            )
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ Memory created successfully!")
            logger.info(f"📄 Response: {json.dumps(result, indent=2)}")
            
            # Wait a bit for processing
            await asyncio.sleep(3)
            
            # Check logs for override application
            logger.info(f"\n🔍 Checking logs for test_id: {test_id}")
            
        else:
            logger.error(f"❌ Request failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_property_overrides())

