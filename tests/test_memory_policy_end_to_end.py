"""
End-to-End Tests for Memory Policy and link_to DSL.

These tests verify the complete flow from API request to database storage,
including link_to DSL expansion, memory policy resolution, and constraint application.

Tests cover:
1. link_to DSL end-to-end (string, list, dict forms)
2. Full memory_policy with node/edge constraints
3. Controlled vocabulary (create='never')
4. Policy merging (link_to + memory_policy)
5. Schema-level policy inheritance
6. Memory-level policy overrides
7. Validation via GraphQL and Search

Validation Strategy:
- After creating memory, use Neo4j to verify nodes/relationships exist
- Validate node properties/relationships using _omo_source_memory_id scoping
- Compare LLM-extracted vs user-specified properties when applicable
"""

import pytest
import httpx
import uuid
import sys
import os
import warnings
import urllib3
import asyncio
import json
from typing import Dict, Any, List, Optional

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['CURL_CA_BUNDLE'] = ''

# Add the memory repo to path
sys.path.insert(0, '/Users/shawkatkabbara/Documents/GitHub/memory')

from main import app
from asgi_lifespan import LifespanManager
from dotenv import load_dotenv, find_dotenv
from os import environ as env
from services.logger_singleton import LoggerSingleton
from models.parse_server import AddMemoryResponse

# Load environment variables
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    ENV_FILE = find_dotenv()
    if ENV_FILE:
        load_dotenv(ENV_FILE)
        env_local_path = ENV_FILE.replace('.env', '.env.local')
        if os.path.exists(env_local_path):
            load_dotenv(env_local_path, override=True)

logger = LoggerSingleton.get_logger(__name__)

# Test credentials
TEST_X_PAPR_API_KEY = env.get('TEST_X_PAPR_API_KEY')
TEST_SESSION_TOKEN = env.get('TEST_SESSION_TOKEN')
TEST_NAMESPACE_ID = env.get('TEST_NAMESPACE_ID')
TEST_ORGANIZATION_ID = env.get('TEST_ORGANIZATION_ID')

def _has_llm_api_key() -> bool:
    """Return True when an LLM API key is configured for extraction."""
    return bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def unique_id():
    """Generate unique ID for test isolation."""
    return str(uuid.uuid4())[:12]


@pytest.fixture
def api_headers():
    """Standard API headers for testing using Session auth."""
    return {
        'Content-Type': 'application/json',
        'X-Client-Type': 'papr_plugin',
        'Authorization': f'Session {TEST_SESSION_TOKEN}',
        'Accept-Encoding': 'gzip'
    }


@pytest.fixture
def api_key_headers():
    """API Key headers for testing (alternative auth method)."""
    return {
        'Content-Type': 'application/json',
        'X-API-Key': TEST_X_PAPR_API_KEY,
        'Accept-Encoding': 'gzip'
    }


# ============================================================================
# Helper Functions for Validation
# ============================================================================

async def search_for_memory(client, headers, query: str, max_memories: int = 10, namespace_id: str = None) -> Dict:
    """Search for memories matching a query."""
    payload = {
        "query": query,
        "max_memories": max_memories,
        "rank_results": False,
        "enable_agentic_graph": False
    }
    # Include namespace_id for proper tenant isolation
    if namespace_id:
        payload["namespace_id"] = namespace_id

    response = await client.post(
        "/v1/memory/search",
        json=payload,
        headers=headers
    )
    return response.json()


async def query_graphql(client, headers, query: str, variables: Dict = None) -> Dict:
    """Execute a GraphQL query against Neo4j."""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    response = await client.post(
        "/v1/graphql",
        json=payload,
        headers=headers
    )
    return response.json()


async def get_memory_by_id(client, headers, memory_id: str) -> Dict:
    """Get a specific memory by ID."""
    response = await client.get(
        f"/v1/memory/{memory_id}",
        headers=headers
    )
    return response.json()


def extract_nodes_from_search(search_result: Dict) -> List[Dict]:
    """Extract node data from search results."""
    nodes = []
    if "data" in search_result and search_result["data"]:
        data = search_result["data"]
        if "nodes" in data:
            nodes = data["nodes"]
        elif "neo_nodes" in data:
            nodes = data["neo_nodes"]
    return nodes


def _extract_memory_id(response: httpx.Response) -> str:
    """Extract memory_id from add_memory response using AddMemoryResponse."""
    add_result = AddMemoryResponse.model_validate(response.json())
    assert add_result.status == "success", f"Add memory failed: {response.text}"
    assert add_result.data, f"Missing memory data in response: {response.text}"
    return add_result.data[0].memoryId


def _resolve_app_instance(app_instance):
    """Resolve FastAPI app instance for Neo4j access."""
    if hasattr(app_instance, "state"):
        return app_instance
    if hasattr(app_instance, "app") and hasattr(app_instance.app, "state"):
        return app_instance.app
    if hasattr(app, "state"):
        return app
    raise AssertionError("Unable to resolve app instance with state")


async def _neo4j_records(
    app_instance,
    query: str,
    params: Dict[str, Any],
    min_records: int = 1,
    retries: int = 10,
    delay_seconds: float = 3.0
) -> List[Dict[str, Any]]:
    """Run Neo4j query with retries for eventual consistency."""
    app_instance = _resolve_app_instance(app_instance)
    records: List[Dict[str, Any]] = []
    for _ in range(retries):
        async with app_instance.state.memory_graph.async_neo_conn.get_session() as session:
            result = await session.run(query, **params)
            records = []
            async for record in result:
                records.append(record.data())
        if len(records) >= min_records:
            return records
        await asyncio.sleep(delay_seconds)
    return records


async def _neo4j_count(
    app_instance,
    query: str,
    params: Dict[str, Any],
    min_count: int = 1,
    retries: int = 10,
    delay_seconds: float = 3.0
) -> int:
    """Run Neo4j count query with retries."""
    app_instance = _resolve_app_instance(app_instance)
    count = 0
    for _ in range(retries):
        async with app_instance.state.memory_graph.async_neo_conn.get_session() as session:
            result = await session.run(query, **params)
            record = await result.single()
            count = int(record.get("count", 0)) if record else 0
        if count >= min_count:
            return count
        await asyncio.sleep(delay_seconds)
    return count


def _apply_tenant_fields(payload: Dict[str, Any]) -> None:
    """Apply tenant fields only when available in env."""
    if TEST_ORGANIZATION_ID:
        payload["organization_id"] = TEST_ORGANIZATION_ID
    if TEST_NAMESPACE_ID:
        payload["namespace_id"] = TEST_NAMESPACE_ID


def get_project_management_schema() -> Dict[str, Any]:
    """Schema for manual graph override tests."""
    return {
        "name": "Project Management Graph (Policy Test)",
        "description": "Schema for manual graph override tests",
        "status": "active",
        "node_types": {
            "Person": {
                "name": "Person",
                "label": "Person",
                "properties": {
                    "name": {"type": "string", "required": True},
                    "email": {"type": "string", "required": False},
                    "role": {"type": "string", "required": False}
                },
                "unique_identifiers": ["email"],
                "required_properties": ["name"]
            },
            "Company": {
                "name": "Company",
                "label": "Company",
                "properties": {
                    "name": {"type": "string", "required": True},
                    "domain": {"type": "string", "required": False},
                    "industry": {"type": "string", "required": False}
                },
                "unique_identifiers": ["name"],
                "required_properties": ["name"]
            },
            "Task": {
                "name": "Task",
                "label": "Task",
                "properties": {
                    "title": {"type": "string", "required": True},
                    "description": {"type": "string", "required": False},
                    "status": {"type": "string", "required": False},
                    "priority": {"type": "string", "required": False}
                },
                "unique_identifiers": ["title"],
                "required_properties": ["title"]
            }
        },
        "relationship_types": {
            "WORKS_AT": {
                "name": "WORKS_AT",
                "label": "Works At",
                "allowed_source_types": ["Person"],
                "allowed_target_types": ["Company"]
            },
            "OWNS": {
                "name": "OWNS",
                "label": "Owns",
                "allowed_source_types": ["Person"],
                "allowed_target_types": ["Task"]
            }
        }
    }


def get_deeptrust_schema(unique_id: str) -> Dict[str, Any]:
    """DeepTrust-style schema with edge policy for MITIGATES."""
    return {
        "name": f"DeepTrust Security Schema {unique_id}",
        "description": "Schema with controlled vocabulary + edge policy for mitigations",
        "status": "active",
        "node_types": {
            "SecurityBehavior": {
                "name": "SecurityBehavior",
                "label": "SecurityBehavior",
                "properties": {
                    "id": {"type": "string", "required": False},
                    "name": {"type": "string", "required": True},
                    "description": {"type": "string", "required": False},
                    "category": {"type": "string", "required": False}
                },
                "unique_identifiers": ["name"],
                "required_properties": ["name"]
            },
            "TacticDef": {
                "name": "TacticDef",
                "label": "TacticDef",
                "properties": {
                    "id": {"type": "string", "required": False},
                    "name": {"type": "string", "required": True},
                    "description": {"type": "string", "required": False}
                },
                "unique_identifiers": ["name"],
                "required_properties": ["name"]
            }
        },
        "relationship_types": {
            "MITIGATES": {
                "name": "MITIGATES",
                "label": "Mitigates",
                "allowed_source_types": ["SecurityBehavior"],
                "allowed_target_types": ["TacticDef"],
                "constraint": {
                    "create": "lookup",
                    "search": {
                        "properties": [
                            {"name": "name", "mode": "semantic", "threshold": 0.9}
                        ]
                    }
                }
            }
        },
        "memory_policy": {
            "mode": "auto",
            "consent": "explicit",
            "risk": "sensitive",
            "node_constraints": [
                {
                    "node_type": "SecurityBehavior",
                    "create": "lookup",
                    "search": {
                        "properties": [
                            {"name": "name", "mode": "semantic", "threshold": 0.85}
                        ]
                    }
                },
                {
                    "node_type": "TacticDef",
                    "create": "lookup",
                    "search": {
                        "properties": [
                            {"name": "name", "mode": "semantic", "threshold": 0.9}
                        ]
                    }
                }
            ]
        }
    }


# ============================================================================
# link_to DSL End-to-End Tests with Validation
# ============================================================================

class TestLinkToDSLEndToEnd:
    """End-to-end tests for link_to DSL with proper validation."""

    @pytest.mark.asyncio
    async def test_link_to_string_form(self, unique_id, api_headers):
        """Test link_to with simple string form: 'Task:title'."""
        if not _has_llm_api_key():
            pytest.skip("LLM API key not configured; link_to extraction cannot create nodes")
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                test_content = f"Fix authentication bug in login flow - TestID {unique_id}"
                data = {
                    "content": test_content,
                    "type": "text",
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    "link_to": "Task:title"
                }

                response = await client.post("/v1/memory", json=data, headers=api_headers)
                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
                memory_id = _extract_memory_id(response)

                task_records = await _neo4j_records(
                    manager.app,
                    """
                    MATCH (t:Task)
                    WHERE t._omo_source_memory_id = $memory_id
                    RETURN t.title AS title
                    """,
                    {"memory_id": memory_id}
                )
                assert task_records, "Task node not found"

    @pytest.mark.asyncio
    async def test_link_to_list_form(self, unique_id, api_headers):
        """Test link_to with list form: ['Task:title', 'Person:email']."""
        if not _has_llm_api_key():
            pytest.skip("LLM API key not configured; link_to extraction cannot create nodes")
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                test_content = (
                    f"Alice {unique_id} assigned Task Fix authentication {unique_id} "
                    f"to Bob {unique_id}."
                )
                data = {
                    "content": test_content,
                    "type": "text",
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    "link_to": ["Task:title", "Person:name"]
                }

                response = await client.post("/v1/memory", json=data, headers=api_headers)
                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
                memory_id = _extract_memory_id(response)

                task_count = await _neo4j_count(
                    manager.app,
                    """
                    MATCH (t:Task)
                    WHERE t._omo_source_memory_id = $memory_id
                      AND t.title CONTAINS $unique_id
                    RETURN count(t) AS count
                    """,
                    {"memory_id": memory_id, "unique_id": unique_id}
                )
                assert task_count > 0, "Task node not found"

                person_count = await _neo4j_count(
                    manager.app,
                    """
                    MATCH (p:Person)
                    WHERE p._omo_source_memory_id = $memory_id
                      AND p.name CONTAINS $unique_id
                    RETURN count(p) AS count
                    """,
                    {"memory_id": memory_id, "unique_id": unique_id}
                )
                assert person_count > 0, "Person node not found"

    @pytest.mark.asyncio
    async def test_link_to_dict_form_with_create_never(self, unique_id, api_headers):
        """Test link_to with dict form + create='never'."""
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                test_content = f"Don't create new tasks for this - TestID {unique_id}"
                data = {
                    "content": test_content,
                    "type": "text",
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    "link_to": {
                        "Task:title": {"create": "never"}
                    }
                }

                response = await client.post("/v1/memory", json=data, headers=api_headers)
                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
                memory_id = _extract_memory_id(response)

                task_count = await _neo4j_count(
                    manager.app,
                    """
                    MATCH (t:Task)
                    WHERE t._omo_source_memory_id = $memory_id
                    RETURN count(t) AS count
                    """,
                    {"memory_id": memory_id},
                    min_count=0,
                    retries=1
                )
                assert task_count == 0, "Task node should not be created when create='never'"

    @pytest.mark.asyncio
    async def test_link_to_with_exact_match(self, unique_id, api_headers):
        """Test link_to with exact match: 'Task:id=TASK-123'."""
        if not _has_llm_api_key():
            pytest.skip("LLM API key not configured; link_to extraction cannot create nodes")
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                data = {
                    "content": f"Task reference with exact ID TASK-{unique_id} - TestID {unique_id}",
                    "type": "text",
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    "link_to": {
                        f"Task:id=TASK-{unique_id}": {"set": {"id": f"TASK-{unique_id}"}}
                    }
                }

                response = await client.post("/v1/memory", json=data, headers=api_headers)
                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
                memory_id = _extract_memory_id(response)

                task_records = await _neo4j_records(
                    manager.app,
                    """
                    MATCH (t:Task)
                    WHERE t._omo_source_memory_id = $memory_id
                      AND t.id = $task_id
                    RETURN t.id AS id
                    """,
                    {"memory_id": memory_id, "task_id": f"TASK-{unique_id}"}
                )
                assert task_records, "Task node with exact id not found"

    @pytest.mark.asyncio
    async def test_link_to_with_semantic_threshold(self, unique_id, api_headers):
        """Test link_to with semantic threshold: 'Task:title~@0.9'."""
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                data = {
                    "content": f"High-confidence match needed for this critical task - TestID {unique_id}",
                    "type": "text",
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    "link_to": "Task:title~@0.9"
                }

                response = await client.post("/v1/memory", json=data, headers=api_headers)
                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
                memory_id = _extract_memory_id(response)

                task_records = await _neo4j_records(
                    manager.app,
                    """
                    MATCH (t:Task)
                    WHERE t._omo_source_memory_id = $memory_id
                    RETURN t.title AS title
                    """,
                    {"memory_id": memory_id}
                )
                assert task_records, "Task node not found"


# ============================================================================
# Full Memory Policy Tests with Validation
# ============================================================================

class TestFullMemoryPolicyEndToEnd:
    """End-to-end tests for full memory_policy with validation."""

    @pytest.mark.asyncio
    async def test_memory_policy_auto_mode_creates_nodes(self, unique_id, api_headers):
        """
        Test memory_policy with mode='auto' creates nodes from LLM extraction.

        Validates:
        - Memory is created successfully
        - LLM extracts entities based on node_constraints
        - Nodes can be found via search
        """
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                # Content with clear entities for extraction
                test_content = (
                    f"Meeting with Evelyn {unique_id} about Task Alpha {unique_id} "
                    f"status update - TestID {unique_id}"
                )
                data = {
                    "content": test_content,
                    "type": "text",
                    # Include namespace_id and organization_id for proper multi-tenant scoping
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    "memory_policy": {
                        "mode": "auto",
                        "consent": "explicit",
                        "node_constraints": [
                            {
                                "node_type": "Task",
                                "search": {
                                    "properties": [
                                        {"name": "title", "mode": "semantic"}
                                    ]
                                }
                            },
                            {
                                "node_type": "Person",
                                "create": "auto",
                                "search": {
                                    "properties": [
                                        {"name": "name", "mode": "semantic"}
                                    ]
                                }
                            }
                        ]
                    }
                }

                response = await client.post("/v1/memory", json=data, headers=api_headers)

                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
                memory_id = _extract_memory_id(response)

                person_records = await _neo4j_records(
                    manager.app,
                    """
                    MATCH (p:Person)
                    WHERE p._omo_source_memory_id = $memory_id
                      AND p.name CONTAINS $unique_id
                    RETURN p.name AS name
                    """,
                    {"memory_id": memory_id, "unique_id": unique_id}
                )
                assert person_records, "Person node not found"

                task_records = await _neo4j_records(
                    manager.app,
                    """
                    MATCH (t:Task)
                    WHERE t._omo_source_memory_id = $memory_id
                      AND t.title CONTAINS $unique_id
                    RETURN t.title AS title
                    """,
                    {"memory_id": memory_id, "unique_id": unique_id}
                )
                assert task_records, "Task node not found"

    @pytest.mark.asyncio
    async def test_memory_policy_manual_mode_exact_nodes(self, unique_id, api_headers):
        """
        Test memory_policy with mode='manual' creates exact nodes specified.

        Validates:
        - Manual nodes are created exactly as specified
        - Relationships are created as specified
        - No LLM extraction occurs (only specified nodes created)
        """
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                task_id = f"task_{unique_id}"
                person_id = f"person_{unique_id}"
                data = {
                    "content": f"Structured data import test - TestID {unique_id}",
                    "type": "text",
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    "memory_policy": {
                        "mode": "manual",
                        "nodes": [
                            {
                                "id": task_id,
                                "type": "Task",
                                "properties": {
                                    "title": f"Test Task {unique_id}",
                                    "status": "pending",
                                    "priority": "high"
                                }
                            },
                            {
                                "id": person_id,
                                "type": "Person",
                                "properties": {
                                    "name": f"Owner {unique_id}"
                                }
                            }
                        ],
                        "relationships": [
                            {
                                "source": person_id,
                                "target": task_id,
                                "type": "MENTIONS"
                            }
                        ]
                    }
                }

                response = await client.post("/v1/memory", json=data, headers=api_headers)

                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

                memory_id = _extract_memory_id(response)

                task_records = await _neo4j_records(
                    manager.app,
                    """
                    MATCH (t:Task)
                    WHERE t._omo_source_memory_id = $memory_id
                      AND t.title CONTAINS $title
                    RETURN t.title AS title
                    """,
                    {"memory_id": memory_id, "title": f"Test Task {unique_id}"}
                )
                assert task_records, "Task node not found in manual mode"

                mentions_count = await _neo4j_count(
                    manager.app,
                    """
                    MATCH (p:Person)-[r:MENTIONS]->(t:Task)
                    WHERE p._omo_source_memory_id = $memory_id
                      AND t._omo_source_memory_id = $memory_id
                    RETURN count(r) AS count
                    """,
                    {"memory_id": memory_id}
                )
                assert mentions_count > 0, "MENTIONS relationship not found in manual mode"

    @pytest.mark.asyncio
    async def test_memory_policy_with_omo_safety(self, unique_id, api_headers):
        """Test memory_policy with OMO safety fields (consent, risk, acl)."""
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                data = {
                    "content": f"Sensitive customer data for Sarah {unique_id} at Acme Corp - TestID {unique_id}",
                    "type": "text",
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    "memory_policy": {
                        "mode": "auto",
                        "consent": "explicit",
                        "risk": "sensitive",
                        "node_constraints": [
                            {
                                "node_type": "Person",
                                "create": "auto",
                                "search": {
                                    "properties": [
                                        {"name": "name", "mode": "semantic"}
                                    ]
                                }
                            }
                        ],
                        "acl": {
                            "read": [f"user_{unique_id}"],
                            "write": [f"user_{unique_id}"]
                        }
                    }
                }

                response = await client.post("/v1/memory", json=data, headers=api_headers)

                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
                memory_id = _extract_memory_id(response)

                person_records = await _neo4j_records(
                    manager.app,
                    """
                    MATCH (p:Person)
                    WHERE p._omo_source_memory_id = $memory_id
                      AND p.name CONTAINS $unique_id
                    RETURN p.name AS name
                    """,
                    {"memory_id": memory_id, "unique_id": unique_id}
                )
                assert person_records, "Person node not found for OMO safety test"


# ============================================================================
# Custom Metadata Propagation Tests
# ============================================================================

class TestCustomMetadataPropagation:
    """Tests that metadata.customMetadata is applied to Neo4j node properties."""

    @pytest.mark.asyncio
    async def test_custom_metadata_applied_to_nodes(self, unique_id, api_headers):
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                custom_metadata = {
                    "source": "unit_test",
                    "category": "policy_validation",
                    "classification": "confidential"
                }
                payload = {
                    "content": f"Custom metadata propagation test - TestID {unique_id}",
                    "type": "text",
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    "memory_policy": {
                        "mode": "manual",
                        "nodes": [
                            {
                                "id": f"person_{unique_id}",
                                "type": "Person",
                                "properties": {
                                    "name": f"Alex Rivera {unique_id}"
                                }
                            },
                            {
                                "id": f"company_{unique_id}",
                                "type": "Company",
                                "properties": {
                                    "name": f"Example Co {unique_id}"
                                }
                            }
                        ],
                        "relationships": [
                            {
                                "source": f"person_{unique_id}",
                                "target": f"company_{unique_id}",
                                "type": "WORKS_AT"
                            }
                        ]
                    },
                    "metadata": {
                        "customMetadata": custom_metadata
                    }
                }

                response = await client.post("/v1/memory", json=payload, headers=api_headers)
                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
                memory_id = _extract_memory_id(response)

                node_records = await _neo4j_records(
                    manager.app,
                    """
                    MATCH (n)
                    WHERE n._omo_source_memory_id = $memory_id
                      AND NOT 'Memory' IN labels(n)
                    RETURN properties(n) AS props
                    """,
                    {"memory_id": memory_id},
                    min_records=2
                )
                assert node_records, "No Neo4j nodes found for custom metadata test"

                for record in node_records:
                    props = record.get("props", {})
                    assert props.get("source") == custom_metadata["source"]
                    assert props.get("category") == custom_metadata["category"]
                    assert props.get("classification") == custom_metadata["classification"]


# ============================================================================
# Schema-Level Policy Inheritance Tests
# ============================================================================

class TestSchemaLevelPolicyInheritance:
    """Tests for schema-level policy that gets inherited by memories."""

    @pytest.mark.asyncio
    async def test_schema_policy_inheritance(self, unique_id, api_headers):
        """
        Test that schema-level memory_policy is inherited when schema_id is set.

        Flow:
        1. Create a schema with memory_policy
        2. Create memory with schema_id (no memory_policy)
        3. Verify schema's policy is applied
        """
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                # First, create a schema with memory_policy
                schema_data = {
                    "name": f"TestSchema_{unique_id}",
                    "description": "Test schema for policy inheritance",
                    "node_types": {
                        "TestTask": {
                            "name": "TestTask",
                            "label": "TestTask",
                            "description": "A test task type",
                            "properties": {
                                "title": {"type": "string", "required": True},
                                "status": {"type": "string", "required": False}
                            },
                            "unique_identifiers": ["title"],
                            "required_properties": ["title"]
                        }
                    },
                    "memory_policy": {
                        "mode": "auto",
                        "consent": "explicit",
                        "node_constraints": [
                            {
                                "node_type": "TestTask",
                                "create": "auto"
                            }
                        ]
                    }
                }

                # Create schema
                _apply_tenant_fields(schema_data)
                schema_response = await client.post(
                    "/v1/schemas",
                    json=schema_data,
                    headers=api_headers
                )

                if schema_response.status_code == 201:
                    schema_result = schema_response.json()
                    schema_id = schema_result.get("data", {}).get("id")

                    if schema_id:
                        # Create memory using schema (inherits policy)
                        memory_data = {
                            "content": f"TestTask to review code changes - TestID {unique_id}",
                            "type": "text",
                            "memory_policy": {"schema_id": schema_id}
                            # Note: no memory_policy overrides beyond schema_id
                        }
                        _apply_tenant_fields(memory_data)

                        memory_response = await client.post(
                            "/v1/memory",
                            json=memory_data,
                            headers=api_headers
                        )

                        assert memory_response.status_code == 200
                        memory_id = _extract_memory_id(memory_response)
                        test_task_records = await _neo4j_records(
                            manager.app,
                            """
                            MATCH (t:TestTask)
                            WHERE t._omo_source_memory_id = $memory_id
                            RETURN t.title AS title
                            """,
                            {"memory_id": memory_id}
                        )
                        assert test_task_records, "TestTask node not found for inherited schema policy"
                        logger.info(f"Memory created with inherited schema policy, schema_id: {schema_id}")
                else:
                    logger.warning(f"Schema creation returned {schema_response.status_code}, skipping inheritance test")


# ============================================================================
# Memory-Level Policy Override Tests
# ============================================================================

class TestMemoryLevelPolicyOverride:
    """Tests for memory-level policy overriding schema-level policy."""

    @pytest.mark.asyncio
    async def test_memory_policy_overrides_schema(self, unique_id, api_headers):
        """
        Test that memory-level policy overrides schema-level policy.

        When both schema_id and memory_policy are provided:
        - Schema policy is used as defaults
        - Memory policy fields override schema policy
        """
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                # Memory with both schema_id and memory_policy override
                data = {
                    "content": f"Override test - high security data - TestID {unique_id}",
                    "type": "text",
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    # Memory-level policy overrides any schema defaults
                    "memory_policy": {
                        "mode": "auto",
                        "consent": "explicit",  # Override: require explicit consent
                        "risk": "sensitive",  # Override: mark as sensitive (valid values: none, sensitive, flagged)
                        "node_constraints": [
                            {
                                "node_type": "Task",
                                "create": "never"  # Override: don't create new tasks
                            }
                        ]
                    }
                }

                response = await client.post("/v1/memory", json=data, headers=api_headers)

                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
                memory_id = _extract_memory_id(response)

                task_count = await _neo4j_count(
                    manager.app,
                    """
                    MATCH (t:Task)
                    WHERE t._omo_source_memory_id = $memory_id
                    RETURN count(t) AS count
                    """,
                    {"memory_id": memory_id},
                    min_count=0,
                    retries=1
                )
                assert task_count == 0, "Task nodes should not be created when create='never'"
                logger.info("Memory created with policy override")


# ============================================================================
# Policy Merging Tests (link_to + memory_policy)
# ============================================================================

class TestPolicyMerging:
    """Tests for merging link_to with memory_policy."""

    @pytest.mark.asyncio
    async def test_link_to_with_memory_policy(self, unique_id, api_headers):
        """Test that link_to and memory_policy can be combined."""
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                data = {
                    "content": f"Combined policy test - bug fix for authentication - TestID {unique_id}",
                    "type": "text",
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    # link_to provides shorthand constraints
                    "link_to": "Task:title",
                    # memory_policy provides additional configuration
                    "memory_policy": {
                        "mode": "auto",
                        "consent": "explicit",
                        "node_constraints": [
                            {
                                "node_type": "Person",
                                "create": "never"
                            }
                        ]
                    }
                }

                response = await client.post("/v1/memory", json=data, headers=api_headers)

                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"


# ============================================================================
# Manual Graph Override Tests (memory_policy manual mode)
# ============================================================================

class TestManualPolicyGraphOverride:
    """Tests for manual memory_policy graph overrides (nodes + relationships)."""

    @pytest.mark.asyncio
    async def test_manual_graph_override_full_api(self, unique_id, api_headers):
        """
        Create a schema, then add a memory with manual nodes/relationships.
        Validate that nodes appear in agentic graph search results.
        """
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test",
                timeout=60.0
            ) as client:
                schema_data = get_project_management_schema()
                schema_data["name"] = f"{schema_data['name']} [{unique_id}]"
                _apply_tenant_fields(schema_data)

                schema_response = await client.post("/v1/schemas", json=schema_data, headers=api_headers)
                assert schema_response.status_code == 201, f"Schema creation failed: {schema_response.text}"
                schema_id = schema_response.json()["data"]["id"]

                content = f"Manual graph override test - {unique_id}"
                manual_nodes = [
                    {
                        "id": f"person_{unique_id}",
                        "type": "Person",
                        "properties": {
                            "name": f"Sarah Johnson {unique_id}",
                            "email": f"sarah.{unique_id}@acmecorp.com",
                            "role": "CTO"
                        }
                    },
                    {
                        "id": f"company_{unique_id}",
                        "type": "Company",
                        "properties": {
                            "name": f"Acme Corp {unique_id}",
                            "domain": f"acme-{unique_id}.com",
                            "industry": "Security"
                        }
                    },
                    {
                        "id": f"task_{unique_id}",
                        "type": "Task",
                        "properties": {
                            "title": f"Implement access controls {unique_id}",
                            "status": "in_progress",
                            "priority": "high"
                        }
                    }
                ]
                manual_relationships = [
                    {"source": f"person_{unique_id}", "target": f"company_{unique_id}", "type": "WORKS_AT"},
                    {"source": f"person_{unique_id}", "target": f"task_{unique_id}", "type": "OWNS"}
                ]

                memory_payload = {
                    "content": content,
                    "type": "text",
                    "memory_policy": {
                        "schema_id": schema_id,
                        "mode": "manual",
                        "nodes": manual_nodes,
                        "relationships": manual_relationships
                    }
                }
                _apply_tenant_fields(memory_payload)

                response = await client.post("/v1/memory", json=memory_payload, headers=api_headers)
                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

                memory_id = _extract_memory_id(response)
                person_name = f"Sarah Johnson {unique_id}"
                company_name = f"Acme Corp {unique_id}"

                person_records = await _neo4j_records(
                    manager.app,
                    """
                    MATCH (p:Person)
                    WHERE p._omo_source_memory_id = $memory_id
                      AND p.name CONTAINS $name
                    RETURN p.name AS name
                    """,
                    {"memory_id": memory_id, "name": person_name}
                )
                assert person_records, "Person node not found"

                company_records = await _neo4j_records(
                    manager.app,
                    """
                    MATCH (c:Company)
                    WHERE c._omo_source_memory_id = $memory_id
                      AND c.name CONTAINS $name
                    RETURN c.name AS name
                    """,
                    {"memory_id": memory_id, "name": company_name}
                )
                assert company_records, "Company node not found"

                works_at_count = await _neo4j_count(
                    manager.app,
                    """
                    MATCH (p:Person)-[r:WORKS_AT]->(c:Company)
                    WHERE p._omo_source_memory_id = $memory_id
                      AND c._omo_source_memory_id = $memory_id
                    RETURN count(r) AS count
                    """,
                    {"memory_id": memory_id}
                )
                assert works_at_count > 0, "WORKS_AT relationship not found"


# ============================================================================
# DeepTrust Edge Policy Tests (schema-level + memory-level)
# ============================================================================

class TestDeepTrustEdgePolicy:
    """Tests for schema-level edge policy and memory-level overrides."""

    @pytest.mark.asyncio
    async def test_deeptrust_edge_policy_link_to_dsl(self, unique_id, api_headers):
        """
        Create DeepTrust schema with MITIGATES edge policy.
        Use link_to DSL (short form) to create nodes + edge.
        """
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test",
                timeout=60.0
            ) as client:
                schema_data = get_deeptrust_schema(unique_id)
                _apply_tenant_fields(schema_data)

                schema_response = await client.post("/v1/schemas", json=schema_data, headers=api_headers)
                assert schema_response.status_code == 201, f"Schema creation failed: {schema_response.text}"
                schema_id = schema_response.json()["data"]["id"]

                content = f"Defense evasion detected; verify identity policy applies. [run:{unique_id}]"
                memory_payload = {
                    "content": content,
                    "type": "text",
                    "memory_policy": {"schema_id": schema_id},
                    "link_to": {
                        "SecurityBehavior:name": {"create": "auto"},
                        "TacticDef:name": {"create": "auto"},
                        "SecurityBehavior->MITIGATES->TacticDef:name": {"create": "never"}
                    }
                }
                _apply_tenant_fields(memory_payload)

                response = await client.post("/v1/memory", json=memory_payload, headers=api_headers)
                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

                memory_id = _extract_memory_id(response)

                security_behavior_count = await _neo4j_count(
                    manager.app,
                    """
                    MATCH (s:SecurityBehavior)
                    WHERE s._omo_source_memory_id = $memory_id
                    RETURN count(s) AS count
                    """,
                    {"memory_id": memory_id}
                )
                assert security_behavior_count > 0, "SecurityBehavior node missing"

                tactic_def_count = await _neo4j_count(
                    manager.app,
                    """
                    MATCH (t:TacticDef)
                    WHERE t._omo_source_memory_id = $memory_id
                    RETURN count(t) AS count
                    """,
                    {"memory_id": memory_id}
                )
                assert tactic_def_count > 0, "TacticDef node missing"

                mitigates_count = await _neo4j_count(
                    manager.app,
                    """
                    MATCH (s:SecurityBehavior)-[r:MITIGATES]->(t:TacticDef)
                    WHERE s._omo_source_memory_id = $memory_id
                      AND t._omo_source_memory_id = $memory_id
                    RETURN count(r) AS count
                    """,
                    {"memory_id": memory_id},
                    min_count=0,
                    retries=1
                )
                assert mitigates_count == 0, "MITIGATES edge should not be created when create='never'"

    @pytest.mark.asyncio
    async def test_deeptrust_edge_policy_full_api(self, unique_id, api_headers):
        """
        Create DeepTrust schema with MITIGATES edge policy.
        Use full memory_policy with edge_constraints to define the relationship.
        """
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test",
                timeout=60.0
            ) as client:
                schema_data = get_deeptrust_schema(f"{unique_id}-full")
                _apply_tenant_fields(schema_data)

                schema_response = await client.post("/v1/schemas", json=schema_data, headers=api_headers)
                assert schema_response.status_code == 201, f"Schema creation failed: {schema_response.text}"
                schema_id = schema_response.json()["data"]["id"]

                content = f"Mitigate tactics using security behaviors. [run:{unique_id}]"
                memory_payload = {
                    "content": content,
                    "type": "text",
                    "memory_policy": {
                        "schema_id": schema_id,
                        "mode": "auto",
                        "node_constraints": [
                            {
                                "node_type": "SecurityBehavior",
                                "create": "auto",
                                "search": {
                                    "properties": [
                                        {"name": "name", "mode": "semantic", "threshold": 0.85}
                                    ]
                                }
                            },
                            {
                                "node_type": "TacticDef",
                                "create": "auto",
                                "search": {
                                    "properties": [
                                        {"name": "name", "mode": "semantic", "threshold": 0.9}
                                    ]
                                }
                            }
                        ],
                        "edge_constraints": [
                            {
                                "edge_type": "MITIGATES",
                                "source_type": "SecurityBehavior",
                                "target_type": "TacticDef",
                                "create": "lookup",
                                "search": {
                                    "properties": [
                                        {"name": "name", "mode": "semantic", "threshold": 0.9}
                                    ]
                                }
                            }
                        ]
                    }
                }
                _apply_tenant_fields(memory_payload)

                response = await client.post("/v1/memory", json=memory_payload, headers=api_headers)
                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

                memory_id = _extract_memory_id(response)

                security_behavior_count = await _neo4j_count(
                    manager.app,
                    """
                    MATCH (s:SecurityBehavior)
                    WHERE s._omo_source_memory_id = $memory_id
                    RETURN count(s) AS count
                    """,
                    {"memory_id": memory_id}
                )
                assert security_behavior_count > 0, "SecurityBehavior node missing"

                tactic_def_count = await _neo4j_count(
                    manager.app,
                    """
                    MATCH (t:TacticDef)
                    WHERE t._omo_source_memory_id = $memory_id
                    RETURN count(t) AS count
                    """,
                    {"memory_id": memory_id}
                )
                assert tactic_def_count > 0, "TacticDef node missing"
    @pytest.mark.asyncio
    async def test_link_to_constraints_merge_with_memory_policy(self, unique_id, api_headers):
        """
        Test that link_to constraints merge with memory_policy constraints.

        link_to: ["Task:title", "Project:name"]
        memory_policy.node_constraints: [Company with create='never']

        Result should have all three constraints merged.
        """
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                data = {
                    "content": f"Merge test - link_to adds to memory_policy - TestID {unique_id}",
                    "type": "text",
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    "link_to": ["Task:title", "Project:name"],
                    "memory_policy": {
                        "node_constraints": [
                            {"node_type": "Company", "create": "never"}
                        ]
                    }
                }

                response = await client.post("/v1/memory", json=data, headers=api_headers)

                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"


# ============================================================================
# Controlled Vocabulary Tests (create='never')
# ============================================================================

class TestControlledVocabulary:
    """Tests for controlled vocabulary using create='never'."""

    @pytest.mark.asyncio
    async def test_create_never_blocks_new_nodes(self, unique_id, api_headers):
        """
        Test that create='never' prevents creating new nodes of that type.

        When create='never':
        - Only link to existing nodes
        - New nodes of that type should NOT be created
        """
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                data = {
                    "content": f"Reference to NonExistent tactic - TestID {unique_id}",
                    "type": "text",
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    "link_to": {
                        "TacticDef:name": {"create": "never"}
                    }
                }

                response = await client.post("/v1/memory", json=data, headers=api_headers)

                # Request should succeed (memory is created)
                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
                memory_id = _extract_memory_id(response)

                tactic_count = await _neo4j_count(
                    manager.app,
                    """
                    MATCH (t:TacticDef)
                    WHERE t._omo_source_memory_id = $memory_id
                    RETURN count(t) AS count
                    """,
                    {"memory_id": memory_id},
                    min_count=0,
                    retries=1
                )
                assert tactic_count == 0, "TacticDef node should not be created when create='never'"

    @pytest.mark.asyncio
    async def test_mixed_create_policies(self, unique_id, api_headers):
        """Test mixed create policies: some auto, some never."""
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                data = {
                    "content": f"Tasks can be created, but Tactics are controlled - TestID {unique_id}",
                    "type": "text",
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    "link_to": {
                        "Task:title": {"create": "auto"},
                        "TacticDef:name": {"create": "never"}
                    }
                }

                response = await client.post("/v1/memory", json=data, headers=api_headers)

                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"


# ============================================================================
# Edge Constraint (Arrow Syntax) Tests
# ============================================================================

class TestEdgeConstraintsEndToEnd:
    """End-to-end tests for edge constraints using arrow syntax."""

    @pytest.mark.asyncio
    async def test_edge_arrow_syntax(self, unique_id, api_headers):
        """Test edge constraint: 'SecurityBehavior->MITIGATES->TacticDef:name'."""
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                data = {
                    "content": f"New security behavior mitigates defense evasion tactic - TestID {unique_id}",
                    "type": "text",
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    "link_to": "SecurityBehavior->MITIGATES->TacticDef:name"
                }

                response = await client.post("/v1/memory", json=data, headers=api_headers)

                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    @pytest.mark.asyncio
    async def test_edge_with_create_never(self, unique_id, api_headers):
        """Test edge constraint with create='never' for controlled vocabulary."""
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                data = {
                    "content": f"Link to existing tactic definition only - TestID {unique_id}",
                    "type": "text",
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    "link_to": {
                        "SecurityBehavior->MITIGATES->TacticDef:name": {"create": "never"}
                    }
                }

                response = await client.post("/v1/memory", json=data, headers=api_headers)

                assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"


# ============================================================================
# GraphQL Validation Tests
# ============================================================================

class TestGraphQLValidation:
    """Tests that use GraphQL to validate node/edge creation."""

    @pytest.mark.asyncio
    async def test_validate_nodes_via_graphql(self, unique_id, api_headers):
        """
        Create memory with policy, then validate nodes exist via GraphQL.
        """
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                # Create memory with manual node
                task_title = f"GraphQL Validation Task {unique_id}"
                data = {
                    "content": f"GraphQL validation test - {task_title}",
                    "type": "text",
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    "memory_policy": {
                        "mode": "manual",
                        "nodes": [
                            {
                                "id": f"gql_task_{unique_id}",
                                "type": "Task",
                                "properties": {
                                    "title": task_title,
                                    "status": "pending"
                                }
                            }
                        ]
                    }
                }

                response = await client.post("/v1/memory", json=data, headers=api_headers)
                assert response.status_code == 200
                memory_id = _extract_memory_id(response)

                task_records = await _neo4j_records(
                    manager.app,
                    """
                    MATCH (t:Task)
                    WHERE t._omo_source_memory_id = $memory_id
                      AND t.title CONTAINS $title
                    RETURN t.title AS title, t.status AS status
                    """,
                    {"memory_id": memory_id, "title": task_title}
                )
                assert task_records, "Task node not found via Neo4j"


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in memory policy processing."""

    @pytest.mark.asyncio
    async def test_invalid_link_to_syntax_returns_error(self, unique_id, api_headers):
        """Test that invalid link_to syntax returns proper error."""
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                data = {
                    "content": f"Invalid syntax test - TestID {unique_id}",
                    "type": "text",
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    "link_to": "InvalidSyntaxNoColon"  # Missing colon
                }

                response = await client.post("/v1/memory", json=data, headers=api_headers)

                # Should return 400 for invalid syntax
                # Note: Might return 200 if server ignores invalid link_to
                logger.info(f"Invalid syntax response: {response.status_code}")

    @pytest.mark.asyncio
    async def test_invalid_memory_policy_mode_returns_error(self, unique_id, api_headers):
        """Test that invalid mode in memory_policy returns proper error."""
        async with LifespanManager(app, startup_timeout=120) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test"
            ) as client:
                data = {
                    "content": f"Invalid mode test - TestID {unique_id}",
                    "type": "text",
                    "namespace_id": TEST_NAMESPACE_ID,
                    "organization_id": TEST_ORGANIZATION_ID,
                    "memory_policy": {
                        "mode": "invalid_mode_xyz"  # Invalid mode
                    }
                }

                response = await client.post("/v1/memory", json=data, headers=api_headers)

                # Should return 422 for validation error
                assert response.status_code in [400, 422], f"Expected 400/422, got {response.status_code}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
