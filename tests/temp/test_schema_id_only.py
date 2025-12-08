#!/usr/bin/env python3
"""
Focused test: Only add schema_id memory and verify it deeply
"""

import asyncio
import httpx
import os
import time
from pathlib import Path
from neo4j import GraphDatabase
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_API_KEY = os.getenv("PAPR_API_KEY", "f80c5a2940f21882420b41690522cb2c")

# Neo4j connection
NEO4J_URL = os.getenv('NEO4J_URL')
NEO4J_SECRET = os.getenv('NEO4J_SECRET')

# Test state
_test_state = {
    "schema_id": None,
    "memory_id": None,
}


def get_security_schema_data() -> Dict[str, Any]:
    """Get the security schema definition"""
    return {
        "name": "Security Workflow and Risk Detection Schema",
        "description": "Comprehensive ontology to detect security behaviors in conversations",
        "status": "active",
        "node_types": {
            "SecurityBehavior": {
                "name": "SecurityBehavior",
                "label": "SecurityBehavior",
                "description": "A suspicious or malicious pattern detected in conversation",
                "properties": {
                    "id": {"type": "string", "required": True},
                    "behavior_id": {"type": "string", "required": False},
                    "name": {"type": "string", "required": True},
                    "description": {"type": "string", "required": False},
                    "severity": {"type": "string", "required": False},
                },
                "required_properties": ["id", "name"],
                "unique_identifiers": ["id"]
            },
            "Tactic": {
                "name": "Tactic",
                "label": "Tactic",
                "description": "MITRE ATT&CK tactic",
                "properties": {
                    "id": {"type": "string", "required": True},
                    "tactic_id": {"type": "string", "required": False},
                    "name": {"type": "string", "required": True},
                    "description": {"type": "string", "required": False},
                },
                "required_properties": ["id", "name"],
                "unique_identifiers": ["id"]
            },
            "Impact": {
                "name": "Impact",
                "label": "Impact",
                "description": "Potential security impact",
                "properties": {
                    "id": {"type": "string", "required": True},
                    "name": {"type": "string", "required": True},
                    "severity": {"type": "string", "required": False},
                },
                "required_properties": ["id", "name"],
                "unique_identifiers": ["id"]
            }
        },
        "relationship_types": {
            "MAPS_TO_TACTIC": {
                "name": "MAPS_TO_TACTIC",
                "label": "MAPS_TO_TACTIC",
                "description": "Security behavior maps to MITRE tactic",
                "allowed_source_types": ["SecurityBehavior"],
                "allowed_target_types": ["Tactic"]
            },
            "HAS_IMPACT": {
                "name": "HAS_IMPACT",
                "label": "HAS_IMPACT",
                "description": "Security behavior has impact",
                "allowed_source_types": ["SecurityBehavior"],
                "allowed_target_types": ["Impact"]
            }
        }
    }


async def create_security_schema():
    """Test 1: Create Security Schema"""
    print("\n" + "="*60)
    print("Test 1: Create Security Schema")
    print("="*60)
    
    schema_data = get_security_schema_data()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/schemas",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
            json=schema_data
        )
        
        if response.status_code == 200:
            result = response.json()
            schema_id = result.get("schema_id")
            _test_state["schema_id"] = schema_id
            print(f"✅ Schema created: {schema_id}")
            return schema_id
        else:
            print(f"❌ Failed to create schema: {response.status_code}")
            print(f"Response: {response.text}")
            return None


async def add_memory_with_schema_id(schema_id: str):
    """Test 2: Add Memory with schema_id - DEEP INVESTIGATION"""
    print("\n" + "="*60)
    print("Test 2: Add Memory with schema_id - DEEP INVESTIGATION")
    print("="*60)
    
    memory_data = {
        "content": "Security incident detected: SQL injection attempt targeting /api/users endpoint from IP 192.168.1.100. This is a credential access tactic with high severity impact on data confidentiality.",
        "type": "text",
        "schema_id": schema_id,  # Top-level field from SchemaSpecificationMixin
        "metadata": {
            "external_user_id": "security_test_user_001",  # Unique ID for this test - in metadata
            "event_type": "security_incident",
            "test_type": "schema_id_approach"
        }
    }
    
    print(f"📝 Memory payload:")
    print(f"   Content: {memory_data['content'][:100]}...")
    print(f"   Schema ID: {schema_id}")
    print(f"   External User ID: {memory_data['metadata']['external_user_id']}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/memory",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
            json=memory_data
        )
        
        if response.status_code == 200:
            result = response.json()
            memory_id = result.get("memory_id")
            _test_state["memory_id"] = memory_id
            print(f"✅ Memory created with schema_id: {memory_id}")
            print(f"   External User ID: {memory_data['metadata']['external_user_id']}")
            
            # Print full response for investigation
            print(f"\n📊 Full API Response:")
            for key, value in result.items():
                if key == "memory_id":
                    print(f"   {key}: {value}")
                elif isinstance(value, str) and len(value) > 100:
                    print(f"   {key}: {value[:100]}...")
                else:
                    print(f"   {key}: {value}")
            
            return memory_id
        else:
            print(f"❌ Failed to create memory: {response.status_code}")
            print(f"Response: {response.text}")
            return None


def query_neo4j_for_user(external_user_id: str, workspace_id: str = "4YVBwQbdfP") -> Dict[str, Any]:
    """Query Neo4j directly for nodes accessible by external_user_id"""
    if not NEO4J_URL or not NEO4J_SECRET:
        print("❌ Neo4j credentials not found in environment")
        return {"nodes": [], "relationships": []}
    
    driver = GraphDatabase.driver(NEO4J_URL, auth=("neo4j", NEO4J_SECRET))
    
    try:
        with driver.session() as session:
            # Updated query to match external_user_read_access array
            query = """
            MATCH (n)
            WHERE $external_user_id IN n.external_user_read_access 
              AND n.workspace_id = $workspace_id
            OPTIONAL MATCH (n)-[r]-(m)
            WHERE $external_user_id IN m.external_user_read_access 
              AND m.workspace_id = $workspace_id
            RETURN n, 
                   labels(n) as node_labels, 
                   n.external_user_read_access as external_user_read_access,
                   n.user_id as user_id,
                   n.workspace_id as workspace_id,
                   collect(DISTINCT {
                       rel: r, 
                       target: m, 
                       target_labels: labels(m)
                   }) as relationships
            """
            
            print(f"🔍 Neo4j Query:")
            print(f"   External User ID: {external_user_id}")
            print(f"   Workspace ID: {workspace_id}")
            print(f"   Query: {query}")
            
            result = session.run(query, {
                "external_user_id": external_user_id,
                "workspace_id": workspace_id
            })
            
            nodes = []
            all_relationships = []
            
            for record in result:
                if record["n"] is not None:
                    node_data = dict(record["n"])
                    node_data["labels"] = record["node_labels"]
                    node_data["external_user_read_access"] = record["external_user_read_access"]
                    node_data["user_id"] = record["user_id"]
                    node_data["workspace_id"] = record["workspace_id"]
                    nodes.append(node_data)
                    
                    # Process relationships
                    for rel_data in record["relationships"]:
                        if rel_data["rel"] is not None and rel_data["target"] is not None:
                            all_relationships.append({
                                "type": type(rel_data["rel"]).__name__,
                                "properties": dict(rel_data["rel"]),
                                "target_labels": rel_data["target_labels"]
                            })
            
            print(f"📊 Neo4j Query Results:")
            print(f"   Total nodes found: {len(nodes)}")
            print(f"   Total relationships found: {len(all_relationships)}")
            
            if nodes:
                # Group by node type
                node_types = {}
                for node in nodes:
                    for label in node["labels"]:
                        node_types[label] = node_types.get(label, 0) + 1
                
                print(f"   Node types: {', '.join(node_types.keys())}")
                for node_type, count in node_types.items():
                    print(f"     {node_type}: {count}")
                
                # Show first few nodes in detail
                print(f"\n📋 First 3 nodes (detailed):")
                for i, node in enumerate(nodes[:3]):
                    print(f"   Node {i+1}:")
                    print(f"     ID: {node.get('id', 'N/A')}")
                    print(f"     Labels: {node['labels']}")
                    print(f"     Name: {node.get('name', 'N/A')}")
                    print(f"     User ID: {node.get('user_id', 'N/A')}")
                    print(f"     External User Read Access: {node.get('external_user_read_access', 'N/A')}")
                    print(f"     Workspace ID: {node.get('workspace_id', 'N/A')}")
                    if node.get('content'):
                        print(f"     Content: {node['content'][:100]}...")
                    print()
            
            if all_relationships:
                # Group by relationship type
                rel_types = {}
                for rel in all_relationships:
                    rel_type = rel["type"]
                    rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
                
                print(f"   Relationship types: {', '.join(rel_types.keys())}")
                for rel_type, count in rel_types.items():
                    print(f"     {rel_type}: {count}")
            
            return {
                "nodes": nodes,
                "relationships": all_relationships,
                "node_count": len(nodes),
                "relationship_count": len(all_relationships)
            }
            
    except Exception as e:
        print(f"❌ Neo4j query failed: {e}")
        return {"nodes": [], "relationships": [], "error": str(e)}
    finally:
        driver.close()


async def wait_for_processing(seconds: int = 45):
    """Wait for background processing"""
    print("\n" + "="*60)
    print("Test 3: Wait for Background Processing")
    print("="*60)
    print(f"⏳ Waiting {seconds} seconds for background processing...")
    
    for i in range(1, seconds + 1):
        if i % 10 == 0:
            print(f"   ⏳ {i}s elapsed...")
        await asyncio.sleep(1)
    
    print(f"✅ Waited {seconds} seconds")


async def verify_neo4j_storage():
    """Test 4: Deep verification of Neo4j storage"""
    print("\n" + "="*60)
    print("Test 4: DEEP Neo4j Storage Verification")
    print("="*60)
    
    external_user_id = "security_test_user_001"
    workspace_id = "4YVBwQbdfP"  # From logs
    
    print(f"🔍 Deep investigation for: {external_user_id}")
    
    result = query_neo4j_for_user(external_user_id, workspace_id)
    
    if result.get("error"):
        print(f"❌ Neo4j query error: {result['error']}")
        return False
    
    node_count = result["node_count"]
    rel_count = result["relationship_count"]
    
    if node_count > 0:
        print(f"✅ Found {node_count} nodes and {rel_count} relationships")
        
        # Look for specific security schema nodes
        security_nodes = []
        memory_nodes = []
        other_nodes = []
        
        for node in result["nodes"]:
            labels = node["labels"]
            if any(label in ["SecurityBehavior", "Tactic", "Impact"] for label in labels):
                security_nodes.append(node)
            elif "Memory" in labels:
                memory_nodes.append(node)
            else:
                other_nodes.append(node)
        
        print(f"\n📊 Node Analysis:")
        print(f"   Security Schema Nodes: {len(security_nodes)}")
        print(f"   Memory Nodes: {len(memory_nodes)}")
        print(f"   Other Nodes: {len(other_nodes)}")
        
        if security_nodes:
            print(f"\n🔒 Security Schema Nodes Details:")
            for i, node in enumerate(security_nodes[:5]):  # Show first 5
                print(f"   Node {i+1}: {node['labels']} - {node.get('name', 'No name')}")
                if node.get('description'):
                    print(f"     Description: {node['description'][:100]}...")
        
        if memory_nodes:
            print(f"\n💾 Memory Nodes Details:")
            for i, node in enumerate(memory_nodes):
                print(f"   Memory {i+1}: ID={node.get('id', 'N/A')}")
                print(f"     External User ID: {node.get('external_user_id', 'N/A')}")
                print(f"     Test Type: {node.get('test_type', 'N/A')}")
                if node.get('content'):
                    print(f"     Content: {node['content'][:100]}...")
        
        return True
    else:
        print(f"❌ No nodes found for {external_user_id}")
        return False


async def main():
    """Run focused schema_id memory test"""
    print("🚀 Focused Schema ID Memory Test")
    print("="*60)
    print(f"Testing against: {BASE_URL}")
    print(f"Neo4j: {NEO4J_URL}")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Use existing schema ID from successful test
        schema_id = "IeskhPibBx"  # From successful test
        _test_state["schema_id"] = schema_id
        print(f"✅ Using existing schema: {schema_id}")
        
        # Skip schema creation and go directly to memory creation
        
        # Test 2: Add memory with schema_id
        memory_id = await add_memory_with_schema_id(schema_id)
        if not memory_id:
            print("❌ Cannot proceed without memory")
            return
        
        # Test 3: Wait for processing
        await wait_for_processing(45)
        
        # Test 4: Deep Neo4j verification
        success = await verify_neo4j_storage()
        
        duration = time.time() - start_time
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        print(f"Duration: {duration:.2f}s")
        print(f"Schema ID: {_test_state['schema_id']}")
        print(f"Memory ID: {_test_state['memory_id']}")
        
        if success:
            print("✅ Schema ID memory test completed successfully!")
        else:
            print("⚠️  Schema ID memory test completed with issues")
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
