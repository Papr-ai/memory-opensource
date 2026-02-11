"""
Tests for Schema-level Memory Policy functionality.

These tests verify that:
1. Memory policies can be defined at the schema level
2. Schema-level policies are properly resolved when processing memories
3. Memory-level policies correctly override schema-level policies
4. Node constraints are properly merged from schema and memory levels
"""

import pytest
from typing import Dict, List, Any, Optional

# Import the memory policy resolver functions
import sys
sys.path.insert(0, '/Users/shawkatkabbara/Documents/GitHub/memory')

from services.memory_policy_resolver import (
    merge_memory_policies,
    _merge_node_constraints,
    extract_omo_fields_from_policy,
    should_skip_graph_extraction,
    DEFAULT_CONSENT,
    DEFAULT_RISK,
    DEFAULT_MODE
)


# ============================================================================
# Test Data
# ============================================================================

def create_schema_policy() -> Dict[str, Any]:
    """Create a sample schema-level policy."""
    return {
        "mode": "hybrid",
        "consent": "terms",
        "risk": "none",
        "node_constraints": [
            {
                "node_type": "Task",
                "create": "never",
                "force": {"workspace_id": "ws_default"}
            },
            {
                "node_type": "Project",
                "create": "auto",
                "force": {"org_id": "org_123"}
            }
        ]
    }


def create_memory_policy() -> Dict[str, Any]:
    """Create a sample memory-level policy."""
    return {
        "mode": "auto",
        "consent": "explicit",
        "node_constraints": [
            {
                "node_type": "Task",
                "force": {"priority": "high"}
            },
            {
                "node_type": "Person",
                "create": "never"
            }
        ]
    }


# ============================================================================
# Default Values Tests
# ============================================================================

class TestDefaultValues:
    """Tests for default policy values."""

    def test_merge_with_no_policies_returns_defaults(self):
        """Merging None policies should return system defaults."""
        result = merge_memory_policies(None, None)

        assert result["mode"] == DEFAULT_MODE
        assert result["consent"] == DEFAULT_CONSENT
        assert result["risk"] == DEFAULT_RISK
        assert result["node_constraints"] == []
        assert result["nodes"] is None
        assert result["relationships"] is None

    def test_default_consent_is_implicit(self):
        """Default consent level should be 'implicit'."""
        assert DEFAULT_CONSENT == "implicit"

    def test_default_risk_is_none(self):
        """Default risk level should be 'none'."""
        assert DEFAULT_RISK == "none"

    def test_default_mode_is_auto(self):
        """Default mode should be 'auto'."""
        assert DEFAULT_MODE == "auto"


# ============================================================================
# Schema-level Policy Tests
# ============================================================================

class TestSchemaLevelPolicy:
    """Tests for schema-level policy application."""

    def test_schema_policy_applied_when_no_memory_policy(self):
        """Schema-level policy should be used when no memory-level policy."""
        schema_policy = create_schema_policy()
        result = merge_memory_policies(schema_policy, None)

        assert result["mode"] == "hybrid"
        assert result["consent"] == "terms"
        assert result["risk"] == "none"
        assert len(result["node_constraints"]) == 2

    def test_schema_node_constraints_preserved(self):
        """Schema-level node constraints should be preserved."""
        schema_policy = create_schema_policy()
        result = merge_memory_policies(schema_policy, None)

        # Find Task constraint
        task_constraint = next(
            (c for c in result["node_constraints"] if c["node_type"] == "Task"),
            None
        )
        assert task_constraint is not None
        assert task_constraint["create"] == "never"
        assert task_constraint["force"]["workspace_id"] == "ws_default"


# ============================================================================
# Memory-level Override Tests
# ============================================================================

class TestMemoryLevelOverride:
    """Tests for memory-level policy overriding schema-level."""

    def test_memory_mode_overrides_schema_mode(self):
        """Memory-level mode should override schema-level mode."""
        schema_policy = {"mode": "hybrid"}
        memory_policy = {"mode": "auto"}

        result = merge_memory_policies(schema_policy, memory_policy)
        assert result["mode"] == "auto"

    def test_memory_consent_overrides_schema_consent(self):
        """Memory-level consent should override schema-level consent."""
        schema_policy = {"consent": "terms"}
        memory_policy = {"consent": "explicit"}

        result = merge_memory_policies(schema_policy, memory_policy)
        assert result["consent"] == "explicit"

    def test_memory_risk_overrides_schema_risk(self):
        """Memory-level risk should override schema-level risk."""
        schema_policy = {"risk": "none"}
        memory_policy = {"risk": "flagged"}

        result = merge_memory_policies(schema_policy, memory_policy)
        assert result["risk"] == "flagged"

    def test_memory_acl_overrides_schema(self):
        """Memory-level OMO ACL should override schema-level."""
        schema_policy = {"acl": {"read": ["user_a"], "write": ["user_a"]}}
        memory_policy = {"acl": {"read": ["user_b", "user_c"], "write": ["user_b"]}}

        result = merge_memory_policies(schema_policy, memory_policy)
        assert result["acl"]["read"] == ["user_b", "user_c"]
        assert result["acl"]["write"] == ["user_b"]


# ============================================================================
# Node Constraints Merge Tests
# ============================================================================

class TestNodeConstraintsMerge:
    """Tests for merging node constraints from schema and memory levels."""

    def test_memory_constraint_overrides_same_node_type(self):
        """Memory constraint for same node_type should override schema constraint."""
        schema_constraints = [
            {"node_type": "Task", "create": "never", "force": {"status": "pending"}}
        ]
        memory_constraints = [
            {"node_type": "Task", "create": "auto", "force": {"priority": "high"}}
        ]

        result = _merge_node_constraints(schema_constraints, memory_constraints)

        # Should have one Task constraint (memory-level)
        task_constraints = [c for c in result if c["node_type"] == "Task"]
        assert len(task_constraints) == 1
        assert task_constraints[0]["create"] == "auto"
        assert task_constraints[0]["force"]["priority"] == "high"

    def test_schema_constraints_preserved_for_different_node_types(self):
        """Schema constraints for different node_types should be preserved."""
        schema_constraints = [
            {"node_type": "Task", "create": "never"},
            {"node_type": "Project", "force": {"org_id": "org_123"}}
        ]
        memory_constraints = [
            {"node_type": "Person", "create": "never"}
        ]

        result = _merge_node_constraints(schema_constraints, memory_constraints)

        # Should have all three node types
        node_types = {c["node_type"] for c in result}
        assert node_types == {"Task", "Project", "Person"}

    def test_memory_constraint_added_for_new_node_type(self):
        """Memory constraint for new node_type should be added."""
        schema_constraints = [
            {"node_type": "Task", "create": "auto"}
        ]
        memory_constraints = [
            {"node_type": "Customer", "create": "never"}
        ]

        result = _merge_node_constraints(schema_constraints, memory_constraints)

        assert len(result) == 2
        customer_constraint = next(
            (c for c in result if c["node_type"] == "Customer"),
            None
        )
        assert customer_constraint is not None
        assert customer_constraint["create"] == "never"

    def test_full_policy_merge_with_constraints(self):
        """Full policy merge should properly handle node_constraints."""
        schema_policy = create_schema_policy()
        memory_policy = create_memory_policy()

        result = merge_memory_policies(schema_policy, memory_policy)

        # Memory mode should override
        assert result["mode"] == "auto"

        # Memory consent should override
        assert result["consent"] == "explicit"

        # Schema risk should be preserved (memory didn't specify)
        assert result["risk"] == "none"

        # Node constraints should be merged
        task_constraint = next(
            (c for c in result["node_constraints"] if c["node_type"] == "Task"),
            None
        )
        assert task_constraint is not None
        # Memory-level Task constraint should override
        assert task_constraint["force"]["priority"] == "high"

        # Project constraint from schema should be preserved
        project_constraint = next(
            (c for c in result["node_constraints"] if c["node_type"] == "Project"),
            None
        )
        assert project_constraint is not None

        # Person constraint from memory should be added
        person_constraint = next(
            (c for c in result["node_constraints"] if c["node_type"] == "Person"),
            None
        )
        assert person_constraint is not None


# ============================================================================
# OMO Fields Extraction Tests
# ============================================================================

class TestOMOFieldsExtraction:
    """Tests for extracting OMO fields from resolved policy."""

    def test_extract_omo_fields(self):
        """Should extract consent, risk, and acl from policy."""
        policy = {
            "consent": "explicit",
            "risk": "sensitive",
            "acl": {"read": ["user_a"], "write": ["user_a"]},
            "mode": "auto"  # Should not be included
        }

        result = extract_omo_fields_from_policy(policy)

        assert result["consent"] == "explicit"
        assert result["risk"] == "sensitive"
        assert result["acl"]["read"] == ["user_a"]

    def test_extract_omo_fields_with_defaults(self):
        """Should use defaults for missing OMO fields."""
        policy = {"mode": "auto"}

        result = extract_omo_fields_from_policy(policy)

        assert result["consent"] == DEFAULT_CONSENT
        assert result["risk"] == DEFAULT_RISK
        assert result["acl"] is None


# ============================================================================
# Skip Graph Extraction Tests
# ============================================================================

class TestSkipGraphExtraction:
    """Tests for determining when to skip graph extraction."""

    def test_skip_when_consent_none(self):
        """Should skip graph extraction when consent is 'none'."""
        policy = {"consent": "none"}
        assert should_skip_graph_extraction(policy) is True

    def test_dont_skip_when_consent_explicit(self):
        """Should not skip when consent is 'explicit'."""
        policy = {"consent": "explicit"}
        assert should_skip_graph_extraction(policy) is False

    def test_dont_skip_when_consent_implicit(self):
        """Should not skip when consent is 'implicit'."""
        policy = {"consent": "implicit"}
        assert should_skip_graph_extraction(policy) is False

    def test_dont_skip_when_consent_terms(self):
        """Should not skip when consent is 'terms'."""
        policy = {"consent": "terms"}
        assert should_skip_graph_extraction(policy) is False

    def test_dont_skip_when_no_consent_specified(self):
        """Should not skip when consent is not specified (defaults to implicit)."""
        policy = {"mode": "auto"}
        assert should_skip_graph_extraction(policy) is False


# ============================================================================
# Structured Mode Tests
# ============================================================================

class TestStructuredMode:
    """Tests for structured mode policy handling."""

    def test_structured_mode_with_nodes(self):
        """Structured mode should preserve nodes and relationships."""
        memory_policy = {
            "mode": "structured",
            "nodes": [
                {"id": "node_1", "type": "Task", "properties": {"name": "Test"}}
            ],
            "relationships": [
                {"source": "node_1", "target": "node_2", "type": "RELATED_TO"}
            ]
        }

        result = merge_memory_policies(None, memory_policy)

        assert result["mode"] == "structured"
        assert result["nodes"] is not None
        assert len(result["nodes"]) == 1
        assert result["relationships"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
