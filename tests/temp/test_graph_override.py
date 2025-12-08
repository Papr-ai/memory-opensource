#!/usr/bin/env python3

import os
import json
import httpx
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_graph_override():
    """Test graph_override functionality to ensure it still works after our changes"""
    
    logger.info("🚀 Testing Graph Override")
    
    # Simple content for graph override test
    memory_content = "Testing graph override functionality with manual nodes and relationships."
    
    logger.info(f"📝 Memory content: {memory_content}")
    
    # Manual graph override with specific nodes and relationships (new format)
    graph_override = {
        "nodes": [
            {
                "id": "test_node_123",  # ID at top level now
                "label": "TestNode",
                "properties": {
                    "name": "Graph Override Test Node",
                    "type": "manual_override",
                    "created_by": "test_script"
                }
            },
            {
                "id": "concept_456",  # ID at top level now
                "label": "Concept",
                "properties": {
                    "name": "Graph Override Concept",
                    "category": "testing",
                    "description": "Testing manual graph override functionality"
                }
            }
        ],
        "relationships": [
            {
                "source_node_id": "test_node_123",  # New format
                "target_node_id": "concept_456",    # New format
                "relationship_type": "TESTS",       # New format
                "properties": {
                    "test_type": "graph_override",
                    "created_at": datetime.now().isoformat()
                }
            }
        ]
    }
    
    logger.info(f"🔧 Graph override: {json.dumps(graph_override, indent=2)}")
    
    # Create memory with graph_override (manual mode)
    test_id = f"graph_override_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    memory_data = {
        "content": memory_content,
        "type": "text",
        "metadata": {
            "source": "graph_override_test",
            "timestamp": datetime.now().isoformat(),
            "test_type": f"graph_override_{test_id}"
        },
        "graph_generation": {
            "mode": "manual",
            "manual": graph_override  # This should bypass LLM generation
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
            logger.info("✅ Graph override memory created successfully!")
            logger.info(f"📄 Response: {json.dumps(response.json(), indent=2)}")
            
            logger.info(f"\n🔍 Checking logs for test_id: {test_id}")
            logger.info("Look for:")
            logger.info("  - Manual graph override being used (should bypass LLM)")
            logger.info("  - TestNode and Concept nodes being created")
            logger.info("  - TESTS relationship being created")
            logger.info("  - No property overrides or schema selection (manual mode)")
            
        else:
            logger.error(f"❌ Request failed with status {response.status_code}")
            logger.error(f"Response: {response.text}")
            
    except Exception as e:
        logger.error(f"❌ Error making request: {e}")

if __name__ == "__main__":
    test_graph_override()
