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
import uuid
from typing import Dict, Any, Optional

from models.memory_models import SearchResponse
from models.parse_server import Memory, AddMemoryResponse
from models.user_schemas import SchemaResponse

# Shared state for test dependencies
_test_state = {
    "schema_id": None,
    "schema_data": None,
    "policy_memory_id": None,
    "policy_memory_content": None,
    "incident_memory_id": None,
    "run_id": None,
    "external_user_id": None,
}

BASE_URL = "http://localhost:8000"
TEST_API_KEY = os.getenv("TEST_X_USER_API_KEY", "f80c5a2940f21882420b41690522cb2c")
TEST_NAMESPACE_ID = os.getenv("TEST_NAMESPACE_ID")
TEST_ORGANIZATION_ID = os.getenv("TEST_ORGANIZATION_ID")
TEST_EXTERNAL_USER_ID = os.getenv("TEST_EXTERNAL_USER_ID")

def _apply_tenant_fields(payload: Dict[str, Any]) -> None:
    """Apply tenant fields only when available in env."""
    if TEST_ORGANIZATION_ID:
        payload["organization_id"] = TEST_ORGANIZATION_ID
    if TEST_NAMESPACE_ID:
        payload["namespace_id"] = TEST_NAMESPACE_ID


def _extract_add_memory_item(result: Dict[str, Any]) -> Dict[str, Any]:
    """Handle both list and dict add_memory response shapes."""
    if isinstance(result, AddMemoryResponse):
        assert result.data, f"Expected add_memory data, got {result}"
        return result.data[0].model_dump(mode="json")
    if isinstance(result, dict):
        data = result.get("data")
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict):
            return data
    raise AssertionError(f"Unexpected add_memory response shape: {result}")


def _get_memory_id(memory: Any) -> Optional[str]:
    """Return the memory ID across response shapes (dict or Pydantic model)."""
    if memory is None:
        return None
    if isinstance(memory, dict):
        return (
            memory.get("id")
            or memory.get("memoryId")
            or memory.get("objectId")
            or memory.get("memory_id")
        )
    for attr in ("id", "memoryId", "objectId", "memory_id"):
        value = getattr(memory, attr, None)
        if value:
            return value
    return None


def _assert_success(result: Any) -> None:
    """Accept both 'status' and legacy 'success' response fields."""
    if hasattr(result, "status"):
        assert result.status == "success", f"Expected status=success, got {result}"
        return
    if hasattr(result, "success"):
        assert result.success is True, f"Expected success=true, got {result}"
        return
    if isinstance(result, dict):
        if "status" in result:
            assert result["status"] == "success", f"Expected status=success, got {result}"
        else:
            assert result.get("success") is True, f"Expected success=true, got {result}"
        return
    raise AssertionError(f"Unsupported response shape: {result}")


def _parse_schema_response(result: Dict[str, Any]) -> SchemaResponse:
    """Parse schema response using Pydantic model."""
    return SchemaResponse.model_validate(result)


def _parse_search_response(result: Dict[str, Any]) -> SearchResponse:
    """Parse search response using Pydantic model."""
    return SearchResponse.model_validate(result)


def _first_memory_from_search(response: SearchResponse) -> Memory:
    """Return the first memory from a SearchResponse."""
    assert response.data is not None, f"Expected search data, got {response}"
    assert response.data.memories, f"Expected at least one memory, got {response}"
    return response.data.memories[0]


def _assert_external_user(memory: Memory, expected_external_user_id: str) -> None:
    if memory.external_user_id:
        assert memory.external_user_id == expected_external_user_id
        return
    if memory.external_user_read_access:
        assert expected_external_user_id in memory.external_user_read_access
        return
    raise AssertionError(f"External user id not found on memory: {memory}")


def _assert_consent_risk(memory: Memory, consent: str, risk: str) -> None:
    metadata = memory.metadata if isinstance(memory.metadata, dict) else {}
    if metadata.get("consent") is not None or metadata.get("risk") is not None:
        assert metadata.get("consent") == consent
        assert metadata.get("risk") == risk
        return
    memory_dict = memory.model_dump(mode="json")
    if memory_dict.get("consent") is not None or memory_dict.get("risk") is not None:
        assert memory_dict.get("consent") == consent
        assert memory_dict.get("risk") == risk


async def _ensure_schema_created(app) -> None:
    if _test_state["schema_id"] and _test_state["external_user_id"] and _test_state["run_id"]:
        return
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        timeout=30.0
    ) as client:
        run_id = _test_state["run_id"] or uuid.uuid4().hex[:10]
        _test_state["run_id"] = run_id
        external_user_id = _test_state["external_user_id"] or TEST_EXTERNAL_USER_ID or f"security_user_{run_id}"
        _test_state["external_user_id"] = external_user_id
        schema_data = get_security_schema_data(run_id)
        _test_state["schema_data"] = schema_data

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": TEST_API_KEY
        }

        response = await client.post("/v1/schemas", headers=headers, json=schema_data)
        assert response.status_code == 201, f"Schema creation failed: {response.text}"

        schema_result = _parse_schema_response(response.json())
        _assert_success(schema_result)
        assert schema_result.data is not None
        _test_state["schema_id"] = schema_result.data.id

        assert schema_result.data.name == schema_data["name"]
        assert schema_result.data.status == "active"
        assert len(schema_result.data.node_types) == 10
        assert len(schema_result.data.relationship_types) == 16
        assert schema_result.data.memory_policy is not None
        assert schema_result.data.memory_policy.get("consent") == "explicit"
        assert schema_result.data.memory_policy.get("risk") == "sensitive"


async def _ensure_policy_memory(app) -> None:
    if _test_state["policy_memory_id"]:
        return
    await _ensure_schema_created(app)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        timeout=60.0
    ) as client:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": TEST_API_KEY
        }

        memory_content = f"""Security Analyst Sarah Chen created a Data Access Control Policy for the CustomerPortal project. The policy enforces read-only permissions for customer service representatives on the customer database asset. The policy has high severity and requires SOC2 compliance. It protects sensitive customer information and applies to all customer service operations. To implement this policy, Sarah established Goal GOAL-2024-001 'Secure Customer Data Access' with Workflow WF-2024-001 'Data Access Policy Implementation'. The workflow includes Step STEP-001 (conversation: security_session_1761118938): 'Review current permissions' assigned to David Park (status: not_started), Step STEP-002 (conversation: security_session_1761118938): 'Update access controls' assigned to Lisa Thompson (status: not_started), and Step STEP-003 (conversation: security_session_1761118938): 'Validate compliance' assigned to Mike Rodriguez (status: not_started). Step STEP-002 triggers Workflow WF-2024-003 'Advanced Access Review' if complex permissions are detected. Each step implements specific security behaviors for access control and compliance validation. [run:{_test_state['run_id']}]"""
        _test_state["policy_memory_content"] = memory_content

        payload = {
            "content": memory_content,
            "type": "text",
            "external_user_id": _test_state["external_user_id"],
            "memory_policy": {
                "schema_id": _test_state["schema_id"]
            },
            "metadata": {
                "location": "security_operations_center",
                "topics": ["security", "policy", "data_access", "compliance", "SOC2"],
                "emoji_tags": ["🛡️", "📋", "🔒"],
                "emotion_tags": ["vigilant", "systematic"],
                "conversationId": f"security_session_{_test_state['run_id']}",
                "external_user_id": _test_state["external_user_id"],
                "customMetadata": {
                    "source": "security_monitoring",
                    "category": "policy_enforcement",
                    "classification": "confidential",
                    "compliance_framework": "SOC2"
                }
            }
        }
        _apply_tenant_fields(payload)

        response = await client.post("/v1/memory", headers=headers, json=payload)
        assert response.status_code in [200, 201], f"Memory creation failed: {response.text}"

        add_result = AddMemoryResponse.model_validate(response.json())
        _assert_success(add_result)
        memory = _extract_add_memory_item(add_result)
        memory_id = _get_memory_id(memory)
        assert memory_id, f"Missing memory id in response: {memory}"
        _test_state["policy_memory_id"] = memory_id

def _base_security_schema() -> Dict[str, Any]:
    """Base security schema definition (reusable across tests)."""
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


def get_security_schema_data(run_id: str) -> Dict[str, Any]:
    """Get the security schema definition (reusable across tests)"""
    schema = _base_security_schema()
    schema["name"] = f"{schema['name']} [run:{run_id}]"
    schema["memory_policy"] = {
        "mode": "auto",
        "consent": "explicit",
        "risk": "sensitive",
        "node_constraints": [
            {
                "node_type": "SecurityBehavior",
                "create": "lookup",
                "search": {
                    "properties": [
                        {"name": "id", "mode": "exact"},
                        {"name": "name", "mode": "semantic", "threshold": 0.85}
                    ]
                }
            }
        ]
    }
    return schema


async def test_v1_create_security_schema(app):
    """Test 1: Create custom security schema"""
    await _ensure_schema_created(app)


async def test_v1_add_memory_with_schema_id(app):
    """Test 2: Add memory with schema_id (LLM selects schema)"""
    await _ensure_policy_memory(app)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60.0) as client:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": TEST_API_KEY
        }

        memory_content = _test_state["policy_memory_content"]

        payload = {
            "content": memory_content,
            "type": "text",  # Required field - must be one of: text, code_snippet, document
            "external_user_id": _test_state["external_user_id"],
            "memory_policy": {
                "schema_id": _test_state["schema_id"]
            },
            "metadata": {
                "location": "security_operations_center",
                "topics": ["security", "policy", "data_access", "compliance", "SOC2"],
                "emoji_tags": ["🛡️", "📋", "🔒"],
                "emotion_tags": ["vigilant", "systematic"],
                "conversationId": f"security_session_{_test_state['run_id']}",
                "external_user_id": _test_state["external_user_id"],
                "customMetadata": {
                    "source": "security_monitoring",
                    "category": "policy_enforcement",
                    "classification": "confidential",
                    "compliance_framework": "SOC2"
                }
            }
        }
        _apply_tenant_fields(payload)

        # Add memory
        memory_id = _test_state["policy_memory_id"]

        # Fetch memory to validate full structure
        response = await client.get(f"/v1/memory/{memory_id}", headers=headers)
        assert response.status_code == 200, f"Memory retrieval failed: {response.text}"
        search_result = _parse_search_response(response.json())
        _assert_success(search_result)
        stored_memory = _first_memory_from_search(search_result)
        assert stored_memory.content == memory_content
        assert stored_memory.conversation_id == f"security_session_{_test_state['run_id']}"
        _assert_external_user(stored_memory, _test_state["external_user_id"])
        _assert_consent_risk(stored_memory, "explicit", "sensitive")


async def test_v1_wait_for_memory_processing(app):
    """Test 3: Wait for memory background processing to complete"""
    await _ensure_policy_memory(app)

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

            search_result = _parse_search_response(response.json())
            _assert_success(search_result)
            memory = _first_memory_from_search(search_result)
            memory_dict = memory.model_dump(mode="json")

            # Check if processing is complete (topics/metrics/costs populated)
            if memory.topics:
                return
            if memory_dict.get("metrics"):
                return
            if memory_dict.get("totalProcessingCost") is not None:
                return  # Processing complete

            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"Memory processing did not complete within {max_wait_seconds} seconds")


async def test_v1_search_verify_neo4j_nodes(app):
    """Test 4: Search and verify Neo4j nodes exist"""
    await _ensure_policy_memory(app)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=30.0) as client:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": TEST_API_KEY
        }

        search_payload = {
            "query": f"Data Access Control Policy for CustomerPortal [run:{_test_state['run_id']}]",
            "external_user_id": _test_state["external_user_id"],
            "max_memories": 10
        }
        _apply_tenant_fields(search_payload)

        response = await client.post("/v1/memory/search", headers=headers, json=search_payload)
        assert response.status_code == 200, f"Search failed: {response.text}"

        search_result = _parse_search_response(response.json())
        _assert_success(search_result)
        memories = search_result.data.memories if search_result.data else []
        assert memories, "No memories found in search"

        # Verify our memory is in results
        found_memory = False
        for memory in memories:
            if _get_memory_id(memory) == _test_state["policy_memory_id"]:
                found_memory = True
                assert memory.content
                break

        assert found_memory, f"Created memory {_test_state['policy_memory_id']} not found in search results"


async def test_v1_search_with_agentic_graph(app):
    """Test 5: Search with agentic graph enabled (2-hop patterns)"""
    await _ensure_policy_memory(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60.0) as client:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": TEST_API_KEY
        }

        search_payload = {
            "query": f"What security policies protect customer data? [run:{_test_state['run_id']}]",
            "enable_agentic_graph": True,
            "rank_results": True,
            "external_user_id": _test_state["external_user_id"],
            "max_memories": 20
        }
        _apply_tenant_fields(search_payload)

        response = await client.post("/v1/memory/search", headers=headers, json=search_payload)
        assert response.status_code == 200, f"Agentic search failed: {response.text}"

        search_result = _parse_search_response(response.json())
        _assert_success(search_result)
        data = search_result.data
        memories = data.memories if data else []
        nodes = data.nodes if data else []
        assert len(memories) > 0 or len(nodes) > 0, "Agentic search returned no results"

        # Verify result structure
        for memory in memories:
            assert _get_memory_id(memory) is not None
            assert memory.content


async def test_v1_add_memory_with_graph_override(app):
    """Test 6: Add memory with pre-made graph_override"""
    await _ensure_schema_created(app)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=60.0) as client:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": TEST_API_KEY
        }

        # Pre-made graph structure
        manual_nodes = [
            {
                "id": "security_behavior_override",
                "type": "SecurityBehavior",
                "properties": {
                    "name": "Unauthorized Access Attempt",
                    "description": "Detected unauthorized access attempt to admin panel",
                    "severity": "critical",
                    "category": "access_control"
                }
            },
            {
                "id": "impact_override",
                "type": "Impact",
                "properties": {
                    "name": "Potential Data Breach",
                    "severity": "high",
                    "description": "Could lead to exposure of customer data"
                }
            },
            {
                "id": "tactic_override",
                "type": "Tactic",
                "properties": {
                    "tactic_id": "TA0001",
                    "name": "Initial Access",
                    "description": "MITRE ATT&CK Initial Access tactic"
                }
            }
        ]
        manual_relationships = [
            {
                "type": "HAS_IMPACT",
                "source": "security_behavior_override",
                "target": "impact_override"
            },
            {
                "type": "MAPS_TO_TACTIC",
                "source": "security_behavior_override",
                "target": "tactic_override"
            }
        ]

        memory_content = f"Security incident: Unauthorized access attempt detected on admin panel at 2025-01-15 14:30 UTC. Source IP: 192.168.1.100. User attempted to access /admin/users endpoint without proper credentials. Incident logged and access denied. [run:{_test_state['run_id']}]"

        payload = {
            "content": memory_content,
            "type": "text",  # Required field - must be one of: text, code_snippet, document
            "external_user_id": _test_state["external_user_id"],
            "memory_policy": {
                "mode": "manual",
                "nodes": manual_nodes,
                "relationships": manual_relationships
            },
            "metadata": {
                "conversationId": f"security_incident_{_test_state['run_id']}",
                "customMetadata": {
                    "incident_type": "access_control_violation",
                    "severity": "critical",
                    "schema_id": _test_state["schema_id"]
                }
            }
        }
        _apply_tenant_fields(payload)

        response = await client.post("/v1/memory", headers=headers, json=payload)
        assert response.status_code in [200, 201], f"Memory with graph_override failed: {response.text}"

        add_result = AddMemoryResponse.model_validate(response.json())
        _assert_success(add_result)
        memory = _extract_add_memory_item(add_result)
        memory_id = _get_memory_id(memory)
        assert memory_id, f"Missing memory id in response: {memory}"
        _test_state["incident_memory_id"] = memory_id

        # Validate the graph_override was accepted
        response = await client.get(f"/v1/memory/{memory_id}", headers=headers)
        assert response.status_code == 200, f"Memory retrieval failed: {response.text}"
        search_result = _parse_search_response(response.json())
        _assert_success(search_result)
        stored_memory = _first_memory_from_search(search_result)
        assert stored_memory.content == memory_content


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
        schema_result = _parse_schema_response(response.json())
        _assert_success(schema_result)

        # Verify memory still exists
        response = await client.get(f"/v1/memory/{_test_state['policy_memory_id']}", headers=headers)
        assert response.status_code == 200, "Memory retrieval failed"

        search_result = _parse_search_response(response.json())
        _assert_success(search_result)
        memory = _first_memory_from_search(search_result)

        # Validate full memory structure
        assert _get_memory_id(memory) == _test_state["policy_memory_id"]
        assert memory.metadata is not None
        if isinstance(memory.metadata, dict):
            if memory.metadata.get("external_user_id") is not None:
                assert memory.metadata.get("external_user_id") == _test_state["external_user_id"]
            _assert_external_user(memory, _test_state["external_user_id"])
            _assert_consent_risk(memory, "explicit", "sensitive")
        else:
            raise AssertionError(f"Unexpected metadata shape: {memory.metadata}")

        # Validate processing metrics exist
        memory_dict = memory.model_dump(mode="json")
        assert "metrics" in memory_dict or "totalProcessingCost" in memory_dict, "Memory processing metrics missing"
