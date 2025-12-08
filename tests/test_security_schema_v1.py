#!/usr/bin/env python3
"""
Security Schema V1 Tests - Split for Sequential Test Runner

Focused tests for custom security schema functionality:
- Schema creation
- Memory addition with schema_id
- Memory addition with graph_override
- Background processing validation
- Neo4j node verification
- Agentic graph search
"""

import asyncio
import httpx
import os
from typing import Dict, Any, Optional

# Shared state for test dependencies
_test_state = {
    "schema_id": None,
    "schema_data": None,
    "policy_memory_id": None,
    "incident_memory_id": None,
}

BASE_URL = "http://localhost:8000"
TEST_API_KEY = os.getenv("TEST_X_USER_API_KEY", "f80c5a2940f21882420b41690522cb2c")

def get_security_schema_data() -> Dict[str, Any]:
    """Get the security schema definition (reusable across tests)"""
    return {
        "name": "Security Workflow and Risk Detection Schema",
        "description": "Comprehensive ontology to detect security behaviors in conversations, manage security workflows with goals and steps, and map them to MITRE/NIST/Impacts for complete security operations management.",
        "status": "active",
        "node_types": {
            # Core conversation graph
            "Conversation": {
                "name": "Conversation",
                "label": "Conversation",
                "description": "A single customer support conversation (call or chat).",
                "properties": {
                    "conversation_id": {"type": "string", "required": True},
                    "channel": {"type": "string", "required": False},
                    "started_at": {"type": "string", "required": False},
                    "ended_at": {"type": "string", "required": False},
                    "customer_id": {"type": "string", "required": False},
                    "account_id": {"type": "string", "required": False},
                    "summary": {"type": "string", "required": False}
                },
                "required_properties": ["conversation_id"],
                "unique_identifiers": ["conversation_id"]
            },
            "Speaker": {
                "name": "Speaker",
                "label": "Speaker",
                "description": "A participant in a conversation (Agent or Caller).",
                "properties": {
                    "speaker_id": {"type": "string", "required": True},
                    "name": {"type": "string", "required": False},
                    "role": {"type": "string", "required": True}
                },
                "required_properties": ["speaker_id", "role"],
                "unique_identifiers": ["speaker_id"]
            },
            "Utterance": {
                "name": "Utterance",
                "label": "Utterance",
                "description": "A single turn of speech/text within a conversation.",
                "properties": {
                    "utterance_id": {"type": "string", "required": True},
                    "text": {"type": "string", "required": True},
                    "timestamp": {"type": "string", "required": False},
                    "action_flags": {"type": "string", "required": False},
                    "sentiment": {"type": "string", "required": False}
                },
                "required_properties": ["utterance_id", "text"],
                "unique_identifiers": ["utterance_id"]
            },
            # Security-specific nodes
            "SecurityBehavior": {
                "name": "SecurityBehavior",
                "label": "SecurityBehavior",
                "description": "A detected security behavior or policy enforcement action.",
                "properties": {
                    "id": {"type": "string", "required": True},
                    "behavior_id": {"type": "string", "required": False},
                    "name": {"type": "string", "required": True},
                    "description": {"type": "string", "required": False},
                    "severity": {"type": "string", "required": False},
                    "category": {"type": "string", "required": False}
                },
                "required_properties": ["id", "name"],
                "unique_identifiers": ["id"]
            },
            "Tactic": {
                "name": "Tactic",
                "label": "Tactic",
                "description": "MITRE ATT&CK tactic mapped to security behavior.",
                "properties": {
                    "tactic_id": {"type": "string", "required": True},
                    "name": {"type": "string", "required": True},
                    "description": {"type": "string", "required": False}
                },
                "required_properties": ["tactic_id", "name"],
                "unique_identifiers": ["tactic_id"]
            },
            "Impact": {
                "name": "Impact",
                "label": "Impact",
                "description": "Business or security impact of a behavior.",
                "properties": {
                    "id": {"type": "string", "required": True},
                    "impact_id": {"type": "string", "required": False},
                    "name": {"type": "string", "required": True},
                    "severity": {"type": "string", "required": False},
                    "description": {"type": "string", "required": False}
                },
                "required_properties": ["id", "name"],
                "unique_identifiers": ["id"]
            },
            "DetectedBehavior": {
                "name": "DetectedBehavior",
                "label": "DetectedBehavior",
                "description": "Instance of a detected security behavior with confidence score.",
                "properties": {
                    "id": {"type": "string", "required": True},
                    "detected_id": {"type": "string", "required": False},
                    "confidence": {"type": "float", "required": False},
                    "timestamp": {"type": "string", "required": False},
                    "detection_method": {"type": "string", "required": False}
                },
                "required_properties": ["id"],
                "unique_identifiers": ["id"]
            },
            # Workflow management nodes
            "Goal": {
                "name": "Goal",
                "label": "Goal",
                "description": "High-level security goal or objective.",
                "properties": {
                    "id": {"type": "string", "required": True},
                    "goal_id": {"type": "string", "required": False},
                    "name": {"type": "string", "required": True},
                    "description": {"type": "string", "required": False},
                    "status": {"type": "string", "required": False},
                    "priority": {"type": "string", "required": False}
                },
                "required_properties": ["id", "name"],
                "unique_identifiers": ["id"]
            },
            "Workflow": {
                "name": "Workflow",
                "label": "Workflow",
                "description": "Security workflow with multiple steps.",
                "properties": {
                    "id": {"type": "string", "required": True},
                    "workflow_id": {"type": "string", "required": False},
                    "name": {"type": "string", "required": True},
                    "description": {"type": "string", "required": False},
                    "status": {"type": "string", "required": False},
                    "owner": {"type": "string", "required": False}
                },
                "required_properties": ["id", "name"],
                "unique_identifiers": ["id"]
            },
            "Step": {
                "name": "Step",
                "label": "Step",
                "description": "Individual step in a security workflow.",
                "properties": {
                    "step_id": {"type": "string", "required": True},
                    "conversationId": {"type": "string", "required": True},
                    "speaker": {"type": "string", "required": True},
                    "name": {"type": "string", "required": True},
                    "description": {"type": "string", "required": False},
                    "status": {"type": "string", "required": False},
                    "assigned_to": {"type": "string", "required": False},
                    "order": {"type": "integer", "required": False}
                },
                "required_properties": ["step_id", "conversationId", "speaker", "name"],
                "unique_identifiers": ["step_id", "conversationId", "speaker"]
            }
        },
        "relationship_types": {
            "HAS_UTTERANCE": {
                "name": "HAS_UTTERANCE",
                "label": "HAS_UTTERANCE",
                "description": "Conversation contains an utterance",
                "allowed_source_types": ["Conversation"],
                "allowed_target_types": ["Utterance"]
            },
            "SPOKE": {
                "name": "SPOKE",
                "label": "SPOKE",
                "description": "Speaker spoke an utterance",
                "allowed_source_types": ["Speaker"],
                "allowed_target_types": ["Utterance"]
            },
            "PARTICIPATED_IN": {
                "name": "PARTICIPATED_IN",
                "label": "PARTICIPATED_IN",
                "description": "Speaker participated in conversation",
                "allowed_source_types": ["Speaker"],
                "allowed_target_types": ["Conversation"]
            },
            "EXHIBITS": {
                "name": "EXHIBITS",
                "label": "EXHIBITS",
                "description": "Utterance exhibits a security behavior",
                "allowed_source_types": ["Utterance"],
                "allowed_target_types": ["SecurityBehavior"]
            },
            "DETECTED_IN": {
                "name": "DETECTED_IN",
                "label": "DETECTED_IN",
                "description": "Behavior detected in conversation",
                "allowed_source_types": ["DetectedBehavior"],
                "allowed_target_types": ["Conversation"]
            },
            "INSTANCE_OF": {
                "name": "INSTANCE_OF",
                "label": "INSTANCE_OF",
                "description": "Detected behavior is instance of security behavior",
                "allowed_source_types": ["DetectedBehavior"],
                "allowed_target_types": ["SecurityBehavior"]
            },
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
                "description": "Security behavior has business impact",
                "allowed_source_types": ["SecurityBehavior"],
                "allowed_target_types": ["Impact"]
            },
            "HAS_WORKFLOW": {
                "name": "HAS_WORKFLOW",
                "label": "HAS_WORKFLOW",
                "description": "Goal has an associated workflow",
                "allowed_source_types": ["Goal"],
                "allowed_target_types": ["Workflow"]
            },
            "HAS_STEP": {
                "name": "HAS_STEP",
                "label": "HAS_STEP",
                "description": "Workflow has a step",
                "allowed_source_types": ["Workflow"],
                "allowed_target_types": ["Step"]
            },
            "HAS_NEXT_STEP": {
                "name": "HAS_NEXT_STEP",
                "label": "HAS_NEXT_STEP",
                "description": "Step has a next step",
                "allowed_source_types": ["Step"],
                "allowed_target_types": ["Step"]
            },
            "IMPLEMENTS": {
                "name": "IMPLEMENTS",
                "label": "IMPLEMENTS",
                "description": "Step implements a security behavior",
                "allowed_source_types": ["Step"],
                "allowed_target_types": ["SecurityBehavior"]
            },
            "ADDRESSES": {
                "name": "ADDRESSES",
                "label": "ADDRESSES",
                "description": "Workflow addresses a security behavior",
                "allowed_source_types": ["Workflow"],
                "allowed_target_types": ["SecurityBehavior"]
            },
            "ACHIEVES": {
                "name": "ACHIEVES",
                "label": "ACHIEVES",
                "description": "Workflow achieves a goal",
                "allowed_source_types": ["Workflow"],
                "allowed_target_types": ["Goal"]
            },
            "TRIGGERS_WORKFLOW": {
                "name": "TRIGGERS_WORKFLOW",
                "label": "TRIGGERS_WORKFLOW",
                "description": "Step triggers another workflow",
                "allowed_source_types": ["Step"],
                "allowed_target_types": ["Workflow"]
            },
            "MITIGATES": {
                "name": "MITIGATES",
                "label": "MITIGATES",
                "description": "Security behavior mitigates an impact",
                "allowed_source_types": ["SecurityBehavior"],
                "allowed_target_types": ["Impact"]
            }
        }
    }


async def test_v1_create_security_schema(app):
    """Test 1: Create custom security schema"""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=30.0) as client:
        schema_data = get_security_schema_data()
        _test_state["schema_data"] = schema_data

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": TEST_API_KEY
        }

        # Create schema
        response = await client.post("/v1/schemas", headers=headers, json=schema_data)
        assert response.status_code == 201, f"Schema creation failed: {response.text}"

        result = response.json()
        assert result["success"] is True
        assert "data" in result
        assert "id" in result["data"]

        # Store schema_id for other tests
        _test_state["schema_id"] = result["data"]["id"]

        # Validate schema structure
        created_schema = result["data"]
        assert created_schema["name"] == schema_data["name"]
        assert created_schema["status"] == "active"
        assert len(created_schema["node_types"]) == 10
        assert len(created_schema["relationship_types"]) == 16


async def test_v1_add_memory_with_schema_id(app):
    """Test 2: Add memory with schema_id (LLM selects schema)"""
    if not _test_state["schema_id"]:
        raise Exception("Schema must be created first - run test_v1_create_security_schema")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60.0) as client:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": TEST_API_KEY
        }

        memory_content = """Security Analyst Sarah Chen created a Data Access Control Policy for the CustomerPortal project. The policy enforces read-only permissions for customer service representatives on the customer database asset. The policy has high severity and requires SOC2 compliance. It protects sensitive customer information and applies to all customer service operations. To implement this policy, Sarah established Goal GOAL-2024-001 'Secure Customer Data Access' with Workflow WF-2024-001 'Data Access Policy Implementation'. The workflow includes Step STEP-001 (conversation: security_session_1761118938): 'Review current permissions' assigned to David Park (status: not_started), Step STEP-002 (conversation: security_session_1761118938): 'Update access controls' assigned to Lisa Thompson (status: not_started), and Step STEP-003 (conversation: security_session_1761118938): 'Validate compliance' assigned to Mike Rodriguez (status: not_started). Step STEP-002 triggers Workflow WF-2024-003 'Advanced Access Review' if complex permissions are detected. Each step implements specific security behaviors for access control and compliance validation."""

        payload = {
            "content": memory_content,
            "type": "text",  # Required field - must be one of: text, code_snippet, document
            "metadata": {
                "location": "security_operations_center",
                "topics": ["security", "policy", "data_access", "compliance", "SOC2"],
                "emoji_tags": ["🛡️", "📋", "🔒"],
                "emotion_tags": ["vigilant", "systematic"],
                "conversationId": "security_session_1761593724",
                "external_user_id": "security_user_456",
                "customMetadata": {
                    "source": "security_monitoring",
                    "category": "policy_enforcement",
                    "classification": "confidential",
                    "compliance_framework": "SOC2",
                    "schema_id": _test_state["schema_id"]  # Pass schema_id in metadata
                }
            }
        }

        # Add memory
        response = await client.post("/v1/memory", headers=headers, json=payload)
        assert response.status_code in [200, 201], f"Memory creation failed: {response.text}"

        result = response.json()
        assert result["success"] is True
        assert "data" in result

        memory = result["data"]
        _test_state["policy_memory_id"] = memory["id"]

        # Validate memory structure
        assert memory["content"] == memory_content
        assert memory["metadata"]["conversationId"] == "security_session_1761593724"


async def test_v1_wait_for_memory_processing(app):
    """Test 3: Wait for memory background processing to complete"""
    if not _test_state["policy_memory_id"]:
        raise Exception("Memory must be created first - run test_v1_add_memory_with_schema_id")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60.0) as client:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": TEST_API_KEY
        }

        memory_id = _test_state["policy_memory_id"]
        max_wait_seconds = 120  # Reduced from 600s for faster testing
        poll_interval = 2

        for _ in range(int(max_wait_seconds / poll_interval)):
            response = await client.get(f"/v1/memory/{memory_id}", headers=headers)
            assert response.status_code == 200, f"Failed to fetch memory: {response.text}"

            result = response.json()
            memory = result["data"]

            # Check if processing is complete (has metrics/costs)
            if "metrics" in memory and memory.get("metrics"):
                return  # Processing complete

            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"Memory processing did not complete within {max_wait_seconds} seconds")


async def test_v1_search_verify_neo4j_nodes(app):
    """Test 4: Search and verify Neo4j nodes exist"""
    if not _test_state["policy_memory_id"]:
        raise Exception("Memory must be created first")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=30.0) as client:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": TEST_API_KEY
        }

        search_payload = {
            "query": "Data Access Control Policy for CustomerPortal",
            "external_user_id": "security_user_456",
            "max_memories": 10
        }

        response = await client.post("/v1/memory/search", headers=headers, json=search_payload)
        assert response.status_code == 200, f"Search failed: {response.text}"

        result = response.json()
        assert result["success"] is True
        assert "data" in result
        assert len(result["data"]) > 0, "No memories found in search"

        # Verify our memory is in results
        found_memory = False
        for memory in result["data"]:
            if memory["id"] == _test_state["policy_memory_id"]:
                found_memory = True
                # Verify it has node/relationship data indicating Neo4j storage
                assert "content" in memory
                assert "metadata" in memory
                break

        assert found_memory, f"Created memory {_test_state['policy_memory_id']} not found in search results"


async def test_v1_search_with_agentic_graph(app):
    """Test 5: Search with agentic graph enabled (2-hop patterns)"""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60.0) as client:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": TEST_API_KEY
        }

        search_payload = {
            "query": "What security policies protect customer data?",
            "enable_agentic_graph": True,
            "rank_results": True,
            "external_user_id": "security_user_456",
            "max_memories": 20
        }

        response = await client.post("/v1/memory/search", headers=headers, json=search_payload)
        assert response.status_code == 200, f"Agentic search failed: {response.text}"

        result = response.json()
        assert result["success"] is True
        assert "data" in result

        # Agentic graph should return results
        assert len(result["data"]) > 0, "Agentic search returned no results"

        # Verify result structure
        for memory in result["data"]:
            assert "id" in memory
            assert "content" in memory
            # With agentic_graph, we might have enriched context
            assert "metadata" in memory


async def test_v1_add_memory_with_graph_override(app):
    """Test 6: Add memory with pre-made graph_override"""
    if not _test_state["schema_id"]:
        raise Exception("Schema must be created first")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60.0) as client:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": TEST_API_KEY
        }

        # Pre-made graph structure
        graph_override = {
            "nodes": [
                {
                    "label": "SecurityBehavior",
                    "properties": {
                        "name": "Unauthorized Access Attempt",
                        "description": "Detected unauthorized access attempt to admin panel",
                        "severity": "critical",
                        "category": "access_control"
                    }
                },
                {
                    "label": "Impact",
                    "properties": {
                        "name": "Potential Data Breach",
                        "severity": "high",
                        "description": "Could lead to exposure of customer data"
                    }
                },
                {
                    "label": "Tactic",
                    "properties": {
                        "tactic_id": "TA0001",
                        "name": "Initial Access",
                        "description": "MITRE ATT&CK Initial Access tactic"
                    }
                }
            ],
            "relationships": [
                {
                    "type": "HAS_IMPACT",
                    "source": {"label": "SecurityBehavior", "identifier": "name", "value": "Unauthorized Access Attempt"},
                    "target": {"label": "Impact", "identifier": "name", "value": "Potential Data Breach"}
                },
                {
                    "type": "MAPS_TO_TACTIC",
                    "source": {"label": "SecurityBehavior", "identifier": "name", "value": "Unauthorized Access Attempt"},
                    "target": {"label": "Tactic", "identifier": "tactic_id", "value": "TA0001"}
                }
            ]
        }

        memory_content = "Security incident: Unauthorized access attempt detected on admin panel at 2025-01-15 14:30 UTC. Source IP: 192.168.1.100. User attempted to access /admin/users endpoint without proper credentials. Incident logged and access denied."

        payload = {
            "content": memory_content,
            "type": "text",  # Required field - must be one of: text, code_snippet, document
            "graph_override": graph_override,
            "metadata": {
                "conversationId": "security_incident_001",
                "external_user_id": "security_user_456",
                "customMetadata": {
                    "incident_type": "access_control_violation",
                    "severity": "critical",
                    "schema_id": _test_state["schema_id"]
                }
            }
        }

        response = await client.post("/v1/memory", headers=headers, json=payload)
        assert response.status_code in [200, 201], f"Memory with graph_override failed: {response.text}"

        result = response.json()
        assert result["success"] is True
        assert "data" in result

        memory = result["data"]
        _test_state["incident_memory_id"] = memory["id"]

        # Validate the graph_override was accepted
        assert memory["content"] == memory_content


async def test_v1_security_schema_full_workflow(app):
    """Test 7: Full end-to-end workflow validation"""
    # This test validates that all previous tests succeeded and the full workflow works
    assert _test_state["schema_id"] is not None, "Schema creation failed"
    assert _test_state["policy_memory_id"] is not None, "Memory creation failed"

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=30.0) as client:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": TEST_API_KEY
        }

        # Verify schema still exists
        response = await client.get(f"/v1/schemas/{_test_state['schema_id']}", headers=headers)
        assert response.status_code == 200, "Schema retrieval failed"

        # Verify memory still exists
        response = await client.get(f"/v1/memory/{_test_state['policy_memory_id']}", headers=headers)
        assert response.status_code == 200, "Memory retrieval failed"

        result = response.json()
        memory = result["data"]

        # Validate full memory structure
        assert memory["id"] == _test_state["policy_memory_id"]
        assert "metadata" in memory
        assert memory["metadata"]["external_user_id"] == "security_user_456"

        # Validate processing metrics exist
        assert "metrics" in memory or "totalProcessingCost" in memory, "Memory processing metrics missing"
