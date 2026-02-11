"""
Shared fixtures for memory policy tests.

Provides common test data for:
- Node and edge constraints
- DeepTrust security schema
- Mock memory graph
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════
# Node Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_tactic_def_node() -> Dict[str, Any]:
    """Sample TacticDef node for testing - controlled vocabulary."""
    return {
        "type": "TacticDef",
        "properties": {
            "id": "TA0005",
            "name": "Defense Evasion",
            "description": "Techniques that avoid detection or other defenses",
            "severity": "high"
        }
    }


@pytest.fixture
def sample_security_behavior_node() -> Dict[str, Any]:
    """Sample SecurityBehavior node for testing - controlled vocabulary."""
    return {
        "type": "SecurityBehavior",
        "properties": {
            "id": "SB080",
            "name": "Verify Identity",
            "description": "Agent must verify caller identity through MFA or security questions",
            "category": "access_control",
            "trigger_context": "Defense Evasion",
            "required_action": "Complete identity verification before sensitive operations",
            "severity": "critical"
        }
    }


@pytest.fixture
def sample_caller_tactic_node() -> Dict[str, Any]:
    """Sample CallerTactic node for testing - dynamic entity."""
    return {
        "type": "CallerTactic",
        "properties": {
            "tactic_name": "Claimed lost phone",
            "tactic_id": "TA0005",
            "context": "MFA bypass attempt",
            "timestamp": "2025-11-24T14:05:00Z",
            "severity": "high"
        }
    }


@pytest.fixture
def sample_conversation_node() -> Dict[str, Any]:
    """Sample Conversation node for testing."""
    return {
        "type": "Conversation",
        "properties": {
            "call_id": "call_4492",
            "timestamp": "2025-11-24T14:02:00Z",
            "duration": 300,
            "risk_level": "high",
            "security_score": 35
        }
    }


@pytest.fixture
def sample_violation_node() -> Dict[str, Any]:
    """Sample Violation node for testing."""
    return {
        "type": "Violation",
        "properties": {
            "behavior_id": "SB080",
            "expected_action": "Verify Identity",
            "actual_action": "Bypassed Token",
            "timestamp": "2025-11-24T14:05:30Z",
            "severity": "critical"
        }
    }


# ═══════════════════════════════════════════════════════════════════
# Edge Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_mitigates_edge() -> Dict[str, Any]:
    """Sample MITIGATES edge for testing."""
    return {
        "type": "MITIGATES",
        "source_type": "SecurityBehavior",
        "target_type": "TacticDef",
        "properties": {
            "effectiveness": "high",
            "priority": 1
        }
    }


@pytest.fixture
def sample_is_instance_edge() -> Dict[str, Any]:
    """Sample IS_INSTANCE edge for testing."""
    return {
        "type": "IS_INSTANCE",
        "source_type": "CallerTactic",
        "target_type": "TacticDef",
        "properties": {
            "confidence": 0.95
        }
    }


# ═══════════════════════════════════════════════════════════════════
# Constraint Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def controlled_vocabulary_node_constraint() -> Dict[str, Any]:
    """Node constraint for controlled vocabulary (create='never')."""
    return {
        "node_type": "TacticDef",
        "create": "never",
        "search": {
            "properties": [
                {"name": "id", "mode": "exact"},
                {"name": "name", "mode": "semantic", "threshold": 0.90}
            ]
        }
    }


@pytest.fixture
def auto_create_node_constraint() -> Dict[str, Any]:
    """Node constraint for auto-create entities."""
    return {
        "node_type": "CallerTactic",
        "create": "auto",
        "search": {
            "properties": [
                {"name": "tactic_name", "mode": "semantic", "threshold": 0.85}
            ]
        }
    }


@pytest.fixture
def conditional_node_constraint() -> Dict[str, Any]:
    """Node constraint with 'when' condition."""
    return {
        "node_type": "CallerTactic",
        "create": "auto",
        "when": {"severity": "critical"},
        "set": {"alert_security_team": True}
    }


@pytest.fixture
def via_relationship_node_constraint() -> Dict[str, Any]:
    """Node constraint with via_relationship for graph traversal."""
    return {
        "node_type": "SecurityBehavior",
        "create": "never",
        "search": {
            "properties": [
                {"name": "name", "mode": "semantic", "threshold": 0.85}
            ],
            "via_relationship": [
                {
                    "edge_type": "MITIGATES",
                    "target_type": "TacticDef",
                    "target_search": {
                        "properties": [
                            {"name": "name", "mode": "semantic", "threshold": 0.90}
                        ]
                    },
                    "direction": "outgoing"
                }
            ]
        }
    }


@pytest.fixture
def controlled_vocabulary_edge_constraint() -> Dict[str, Any]:
    """Edge constraint for controlled vocabulary relationships."""
    return {
        "edge_type": "MITIGATES",
        "source_type": "SecurityBehavior",
        "target_type": "TacticDef",
        "create": "never",
        "search": {
            "properties": [
                {"name": "id", "mode": "exact"},
                {"name": "name", "mode": "semantic", "threshold": 0.90}
            ]
        }
    }


@pytest.fixture
def auto_create_edge_constraint() -> Dict[str, Any]:
    """Edge constraint for auto-create relationships."""
    return {
        "edge_type": "IS_INSTANCE",
        "source_type": "CallerTactic",
        "target_type": "TacticDef",
        "create": "auto",
        "search": {
            "properties": [
                {"name": "id", "mode": "exact"},
                {"name": "name", "mode": "semantic", "threshold": 0.90}
            ]
        }
    }


# ═══════════════════════════════════════════════════════════════════
# Schema Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def deeptrust_schema_dict() -> Dict[str, Any]:
    """Full DeepTrust security schema for integration tests."""
    return {
        "name": "deeptrust_security",
        "node_types": {
            "TacticDef": {
                "name": "TacticDef",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string", "required": True},
                    "description": {"type": "string"},
                    "severity": {"type": "string"}
                },
                "constraint": {
                    "create": "never",
                    "search": {
                        "properties": [
                            {"name": "id", "mode": "exact"},
                            {"name": "name", "mode": "semantic", "threshold": 0.90}
                        ]
                    }
                }
            },
            "SecurityBehavior": {
                "name": "SecurityBehavior",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string", "required": True},
                    "description": {"type": "string"},
                    "category": {"type": "string"},
                    "trigger_context": {"type": "string"},
                    "required_action": {"type": "string"},
                    "severity": {"type": "string"}
                },
                "constraint": {
                    "create": "never",
                    "search": {
                        "properties": [
                            {"name": "id", "mode": "exact"},
                            {"name": "name", "mode": "semantic", "threshold": 0.85},
                            {"name": "trigger_context", "mode": "semantic", "threshold": 0.90}
                        ],
                        "via_relationship": [
                            {
                                "edge_type": "MITIGATES",
                                "target_type": "TacticDef",
                                "target_search": {
                                    "properties": [
                                        {"name": "id", "mode": "exact"},
                                        {"name": "name", "mode": "semantic", "threshold": 0.90}
                                    ]
                                }
                            }
                        ]
                    }
                }
            },
            "CallerTactic": {
                "name": "CallerTactic",
                "properties": {
                    "tactic_name": {"type": "string"},
                    "tactic_id": {"type": "string"},
                    "context": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "severity": {"type": "string"}
                },
                "constraint": {
                    "create": "auto"
                }
            },
            "Conversation": {
                "name": "Conversation",
                "properties": {
                    "call_id": {"type": "string", "required": True},
                    "timestamp": {"type": "datetime"},
                    "duration": {"type": "integer"},
                    "risk_level": {"type": "string"},
                    "security_score": {"type": "integer"}
                },
                "constraint": {
                    "create": "auto",
                    "search": {
                        "properties": [
                            {"name": "call_id", "mode": "exact"}
                        ]
                    }
                }
            },
            "Violation": {
                "name": "Violation",
                "properties": {
                    "behavior_id": {"type": "string"},
                    "expected_action": {"type": "string"},
                    "actual_action": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "severity": {"type": "string"}
                },
                "constraint": {
                    "create": "auto"
                }
            }
        },
        "relationship_types": {
            "MITIGATES": {
                "name": "MITIGATES",
                "allowed_source_types": ["SecurityBehavior"],
                "allowed_target_types": ["TacticDef"],
                "constraint": {
                    "create": "never",
                    "search": {
                        "properties": [
                            {"name": "id", "mode": "exact"},
                            {"name": "name", "mode": "semantic", "threshold": 0.90}
                        ]
                    }
                }
            },
            "IS_INSTANCE": {
                "name": "IS_INSTANCE",
                "allowed_source_types": ["CallerTactic"],
                "allowed_target_types": ["TacticDef"],
                "constraint": {
                    "create": "auto"
                }
            },
            "HAS_VIOLATION": {
                "name": "HAS_VIOLATION",
                "allowed_source_types": ["Conversation"],
                "allowed_target_types": ["Violation"]
            }
        },
        "memory_policy": {
            "mode": "auto",
            "consent": "implicit",
            "risk": "none"
        }
    }


@pytest.fixture
def schema_memory_policy() -> Dict[str, Any]:
    """Sample schema-level memory policy."""
    return {
        "mode": "auto",
        "consent": "implicit",
        "risk": "none",
        "node_constraints": [
            {
                "node_type": "TacticDef",
                "create": "never",
                "search": {
                    "properties": [
                        {"name": "id", "mode": "exact"}
                    ]
                }
            }
        ],
        "edge_constraints": [
            {
                "edge_type": "MITIGATES",
                "source_type": "SecurityBehavior",
                "target_type": "TacticDef",
                "create": "never"
            }
        ]
    }


@pytest.fixture
def memory_level_policy_override() -> Dict[str, Any]:
    """Sample memory-level policy that overrides schema."""
    return {
        "node_constraints": [
            {
                "node_type": "TacticDef",
                "create": "auto",  # Override schema's "never"
                "search": {
                    "properties": [
                        {"name": "name", "mode": "semantic", "threshold": 0.95}
                    ]
                }
            }
        ]
    }


# ═══════════════════════════════════════════════════════════════════
# Mock Memory Graph
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_memory_graph():
    """Mock memory graph with search methods."""
    mock = AsyncMock()

    # find_node_by_property - exact match
    async def find_by_property(node_type: str, property_name: str, property_value: Any, context: Optional[Dict] = None):
        # Return existing TacticDef if searching for known ID
        if node_type == "TacticDef" and property_name == "id" and property_value == "TA0005":
            return {
                "id": "existing_tactic_1",
                "type": "TacticDef",
                "properties": {
                    "id": "TA0005",
                    "name": "Defense Evasion",
                    "description": "Techniques that avoid detection",
                    "severity": "high"
                }
            }
        # Return existing SecurityBehavior if searching for known ID
        if node_type == "SecurityBehavior" and property_name == "id" and property_value == "SB080":
            return {
                "id": "existing_behavior_1",
                "type": "SecurityBehavior",
                "properties": {
                    "id": "SB080",
                    "name": "Verify Identity",
                    "category": "access_control"
                }
            }
        return None

    mock.find_node_by_property = AsyncMock(side_effect=find_by_property)

    # find_node_by_semantic_match
    async def find_by_semantic(node_type: str, property_name: str, query_text: str, threshold: float = 0.85, context: Optional[Dict] = None):
        if node_type == "TacticDef" and "defense" in query_text.lower():
            return {
                "id": "existing_tactic_1",
                "type": "TacticDef",
                "properties": {
                    "id": "TA0005",
                    "name": "Defense Evasion",
                    "description": "Techniques that avoid detection"
                }
            }
        if node_type == "SecurityBehavior" and "verify" in query_text.lower():
            return {
                "id": "existing_behavior_1",
                "type": "SecurityBehavior",
                "properties": {
                    "id": "SB080",
                    "name": "Verify Identity"
                }
            }
        return None

    mock.find_node_by_semantic_match = AsyncMock(side_effect=find_by_semantic)

    # find_node_by_fuzzy_match
    mock.find_node_by_fuzzy_match = AsyncMock(return_value=None)

    # find_node_via_relationship
    async def find_via_relationship(node_type: str, edge_type: str, target_node_id: str, direction: str = "outgoing", context: Optional[Dict] = None):
        if edge_type == "MITIGATES" and target_node_id == "existing_tactic_1":
            return {
                "id": "existing_behavior_1",
                "type": "SecurityBehavior",
                "properties": {
                    "id": "SB080",
                    "name": "Verify Identity"
                }
            }
        return None

    mock.find_node_via_relationship = AsyncMock(side_effect=find_via_relationship)

    # get_user_schema_async
    mock.get_user_schema_async = AsyncMock(return_value=None)

    return mock


# ═══════════════════════════════════════════════════════════════════
# Context Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_context() -> Dict[str, Any]:
    """Sample context for condition evaluation."""
    return {
        "user_id": "test_user",
        "workspace_id": "test_workspace",
        "session_id": "test_session",
        "metadata": {
            "source": "call_analyzer",
            "timestamp": "2025-11-24T14:00:00Z"
        }
    }


@pytest.fixture
def extracted_node_properties() -> Dict[str, Any]:
    """Sample extracted properties from LLM."""
    return {
        "id": "TA0005",
        "name": "Defense Evasion",
        "description": "Caller claimed lost phone to bypass MFA",
        "severity": "critical"
    }


@pytest.fixture
def extracted_edge_properties() -> Dict[str, Any]:
    """Sample extracted edge properties from LLM."""
    return {
        "confidence": 0.95,
        "detected_at": "2025-11-24T14:05:00Z"
    }


# ═══════════════════════════════════════════════════════════════════
# When Clause Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def simple_when_condition() -> Dict[str, Any]:
    """Simple when condition - single property match."""
    return {"severity": "critical"}


@pytest.fixture
def and_when_condition() -> Dict[str, Any]:
    """When condition with _and operator."""
    return {
        "_and": [
            {"severity": "critical"},
            {"category": "access_control"}
        ]
    }


@pytest.fixture
def or_when_condition() -> Dict[str, Any]:
    """When condition with _or operator."""
    return {
        "_or": [
            {"severity": "critical"},
            {"severity": "high"}
        ]
    }


@pytest.fixture
def not_when_condition() -> Dict[str, Any]:
    """When condition with _not operator."""
    return {
        "_not": {"status": "completed"}
    }


@pytest.fixture
def complex_nested_when_condition() -> Dict[str, Any]:
    """Complex nested when condition."""
    return {
        "_and": [
            {"severity": "critical"},
            {
                "_or": [
                    {"category": "access_control"},
                    {"category": "data_protection"}
                ]
            },
            {
                "_not": {"acknowledged": True}
            }
        ]
    }


# ═══════════════════════════════════════════════════════════════════
# Link-Only Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def link_only_node_constraint() -> Dict[str, Any]:
    """Node constraint using link_only shorthand."""
    return {
        "node_type": "TacticDef",
        "link_only": True,
        "search": {
            "properties": [
                {"name": "id", "mode": "exact"},
                {"name": "name", "mode": "semantic", "threshold": 0.90}
            ]
        }
    }


@pytest.fixture
def link_only_edge_constraint() -> Dict[str, Any]:
    """Edge constraint using link_only shorthand."""
    return {
        "edge_type": "MITIGATES",
        "source_type": "SecurityBehavior",
        "target_type": "TacticDef",
        "link_only": True,
        "search": {
            "properties": [
                {"name": "name", "mode": "semantic", "threshold": 0.90}
            ]
        }
    }
