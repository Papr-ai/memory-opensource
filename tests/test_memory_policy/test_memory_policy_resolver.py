"""
Tests for memory policy resolver - merging schema and memory policies.

Tests the merge_memory_policies function which handles:
- Schema-level policy defaults
- Memory-level policy overrides
- Node constraint merging with precedence
- Edge constraint merging with composite keys

Run with: pytest tests/test_memory_policy/test_memory_policy_resolver.py -v
"""

import pytest
from services.memory_policy_resolver import (
    merge_memory_policies,
    _merge_node_constraints,
    _merge_edge_constraints,
    extract_type_level_constraints,
    extract_omo_fields_from_policy,
    should_skip_graph_extraction,
    DEFAULT_MODE,
    DEFAULT_CONSENT,
    DEFAULT_RISK
)


class TestMergeMemoryPolicies:
    """Test merging of schema and memory-level policies."""

    def test_no_policies_returns_defaults(self):
        """No schema or memory policy should return system defaults."""
        result = merge_memory_policies(None, None)

        assert result["mode"] == DEFAULT_MODE
        assert result["consent"] == DEFAULT_CONSENT
        assert result["risk"] == DEFAULT_RISK
        assert result["node_constraints"] == []
        assert result["edge_constraints"] == []

    def test_schema_only_applies_schema_values(self):
        """Schema-only should apply schema values."""
        schema_policy = {
            "mode": "structured",
            "consent": "explicit",
            "risk": "low",
            "node_constraints": [
                {"node_type": "TacticDef", "create": "never"}
            ]
        }

        result = merge_memory_policies(schema_policy, None)

        assert result["mode"] == "structured"
        assert result["consent"] == "explicit"
        assert result["risk"] == "low"
        assert len(result["node_constraints"]) == 1
        assert result["node_constraints"][0]["node_type"] == "TacticDef"

    def test_memory_only_applies_memory_values(self):
        """Memory-only should apply memory values on top of defaults."""
        memory_policy = {
            "mode": "auto",
            "node_constraints": [
                {"node_type": "CallerTactic", "create": "auto"}
            ]
        }

        result = merge_memory_policies(None, memory_policy)

        assert result["mode"] == "auto"
        assert result["consent"] == DEFAULT_CONSENT  # Default
        assert len(result["node_constraints"]) == 1
        assert result["node_constraints"][0]["node_type"] == "CallerTactic"

    def test_memory_overrides_schema_mode(self):
        """Memory policy mode should override schema mode."""
        schema_policy = {"mode": "structured"}
        memory_policy = {"mode": "auto"}

        result = merge_memory_policies(schema_policy, memory_policy)

        assert result["mode"] == "auto"

    def test_memory_overrides_schema_consent(self):
        """Memory policy consent should override schema consent."""
        schema_policy = {"consent": "implicit"}
        memory_policy = {"consent": "explicit"}

        result = merge_memory_policies(schema_policy, memory_policy)

        assert result["consent"] == "explicit"

    def test_memory_overrides_schema_risk(self):
        """Memory policy risk should override schema risk."""
        schema_policy = {"risk": "none"}
        memory_policy = {"risk": "low"}

        result = merge_memory_policies(schema_policy, memory_policy)

        assert result["risk"] == "low"

    def test_acl_override(self):
        """Memory ACL should override schema ACL."""
        schema_policy = {"acl": {"read": ["user1"]}}
        memory_policy = {"acl": {"read": ["user2"]}}

        result = merge_memory_policies(schema_policy, memory_policy)

        assert result["acl"] == {"read": ["user2"]}

    def test_nodes_override(self):
        """Memory nodes should override schema nodes."""
        schema_policy = {"nodes": ["Person", "Task"]}
        memory_policy = {"nodes": ["User"]}

        result = merge_memory_policies(schema_policy, memory_policy)

        assert result["nodes"] == ["User"]

    def test_relationships_override(self):
        """Memory relationships should override schema relationships."""
        schema_policy = {"relationships": ["ASSIGNED_TO"]}
        memory_policy = {"relationships": ["MITIGATES"]}

        result = merge_memory_policies(schema_policy, memory_policy)

        assert result["relationships"] == ["MITIGATES"]


class TestMergeNodeConstraints:
    """Test node constraint merging with precedence."""

    def test_empty_constraints_returns_empty(self):
        """Empty constraints should return empty list."""
        result = _merge_node_constraints([], [])
        assert result == []

    def test_schema_only_constraints(self):
        """Schema-only constraints should be preserved."""
        schema_constraints = [
            {"node_type": "TacticDef", "create": "never"},
            {"node_type": "SecurityBehavior", "create": "never"}
        ]

        result = _merge_node_constraints(schema_constraints, [])

        assert len(result) == 2
        assert result[0]["node_type"] == "TacticDef"
        assert result[1]["node_type"] == "SecurityBehavior"

    def test_memory_only_constraints(self):
        """Memory-only constraints should be added."""
        memory_constraints = [
            {"node_type": "CallerTactic", "create": "auto"}
        ]

        result = _merge_node_constraints([], memory_constraints)

        assert len(result) == 1
        assert result[0]["node_type"] == "CallerTactic"

    def test_memory_overrides_schema_same_type(self):
        """Memory constraint should override schema constraint for same node_type."""
        schema_constraints = [
            {"node_type": "TacticDef", "create": "never", "search": {"properties": []}}
        ]
        memory_constraints = [
            {"node_type": "TacticDef", "create": "auto", "search": {"properties": [{"name": "name", "mode": "semantic"}]}}
        ]

        result = _merge_node_constraints(schema_constraints, memory_constraints)

        assert len(result) == 1
        assert result[0]["node_type"] == "TacticDef"
        assert result[0]["create"] == "auto"  # Memory override
        assert len(result[0]["search"]["properties"]) == 1  # Memory search

    def test_memory_adds_new_constraint(self):
        """Memory constraint for new type should be added."""
        schema_constraints = [
            {"node_type": "TacticDef", "create": "never"}
        ]
        memory_constraints = [
            {"node_type": "CallerTactic", "create": "auto"}
        ]

        result = _merge_node_constraints(schema_constraints, memory_constraints)

        assert len(result) == 2
        types = [c["node_type"] for c in result]
        assert "TacticDef" in types
        assert "CallerTactic" in types

    def test_mixed_override_and_preserve(self):
        """Mix of overridden and preserved constraints."""
        schema_constraints = [
            {"node_type": "TacticDef", "create": "never"},
            {"node_type": "SecurityBehavior", "create": "never"}
        ]
        memory_constraints = [
            {"node_type": "TacticDef", "create": "auto"},  # Override
            {"node_type": "Violation", "create": "auto"}    # New
        ]

        result = _merge_node_constraints(schema_constraints, memory_constraints)

        assert len(result) == 3

        tactic_constraint = next(c for c in result if c["node_type"] == "TacticDef")
        assert tactic_constraint["create"] == "auto"  # Overridden

        behavior_constraint = next(c for c in result if c["node_type"] == "SecurityBehavior")
        assert behavior_constraint["create"] == "never"  # Preserved

        violation_constraint = next(c for c in result if c["node_type"] == "Violation")
        assert violation_constraint["create"] == "auto"  # Added

    def test_constraint_without_node_type_ignored(self):
        """Constraint without node_type should be ignored in memory_by_type lookup."""
        schema_constraints = [
            {"node_type": "TacticDef", "create": "never"}
        ]
        memory_constraints = [
            {"create": "auto"}  # No node_type - will be ignored
        ]

        result = _merge_node_constraints(schema_constraints, memory_constraints)

        # Memory constraint without node_type is ignored, only schema constraint preserved
        assert len(result) == 1
        assert result[0]["node_type"] == "TacticDef"
        assert result[0]["create"] == "never"


class TestMergeEdgeConstraints:
    """Test edge constraint merging with composite keys."""

    def test_empty_constraints_returns_empty(self):
        """Empty constraints should return empty list."""
        result = _merge_edge_constraints([], [])
        assert result == []

    def test_schema_only_edge_constraints(self):
        """Schema-only edge constraints should be preserved."""
        schema_constraints = [
            {"edge_type": "MITIGATES", "source_type": "SecurityBehavior", "target_type": "TacticDef", "create": "never"}
        ]

        result = _merge_edge_constraints(schema_constraints, [])

        assert len(result) == 1
        assert result[0]["edge_type"] == "MITIGATES"

    def test_memory_overrides_schema_exact_key(self):
        """Memory should override schema with same composite key."""
        schema_constraints = [
            {"edge_type": "MITIGATES", "source_type": "SecurityBehavior", "target_type": "TacticDef", "create": "never"}
        ]
        memory_constraints = [
            {"edge_type": "MITIGATES", "source_type": "SecurityBehavior", "target_type": "TacticDef", "create": "auto"}
        ]

        result = _merge_edge_constraints(schema_constraints, memory_constraints)

        assert len(result) == 1
        assert result[0]["create"] == "auto"  # Memory override

    def test_different_target_type_not_override(self):
        """Different target_type should not override."""
        schema_constraints = [
            {"edge_type": "MITIGATES", "source_type": "SecurityBehavior", "target_type": "TacticDef", "create": "never"}
        ]
        memory_constraints = [
            {"edge_type": "MITIGATES", "source_type": "SecurityBehavior", "target_type": "Impact", "create": "auto"}
        ]

        result = _merge_edge_constraints(schema_constraints, memory_constraints)

        assert len(result) == 2  # Both constraints preserved

    def test_composite_key_matching(self):
        """Composite key (edge_type, source_type, target_type) should match exactly."""
        schema_constraints = [
            {"edge_type": "MITIGATES", "source_type": "SecurityBehavior", "target_type": "TacticDef", "create": "never"},
            {"edge_type": "IS_INSTANCE", "source_type": "CallerTactic", "target_type": "TacticDef", "create": "auto"}
        ]
        memory_constraints = [
            {"edge_type": "MITIGATES", "source_type": "SecurityBehavior", "target_type": "TacticDef", "create": "auto"}
        ]

        result = _merge_edge_constraints(schema_constraints, memory_constraints)

        assert len(result) == 2

        mitigates = next(c for c in result if c["edge_type"] == "MITIGATES")
        assert mitigates["create"] == "auto"  # Overridden

        is_instance = next(c for c in result if c["edge_type"] == "IS_INSTANCE")
        assert is_instance["create"] == "auto"  # Preserved

    def test_edge_type_only_key(self):
        """Edge constraint with only edge_type should use (edge_type, None, None) as key."""
        schema_constraints = [
            {"edge_type": "MITIGATES", "create": "never"}
        ]
        memory_constraints = [
            {"edge_type": "MITIGATES", "create": "auto"}
        ]

        result = _merge_edge_constraints(schema_constraints, memory_constraints)

        assert len(result) == 1
        assert result[0]["create"] == "auto"


class TestExtractTypeLevelConstraints:
    """Test extraction of constraints from schema type definitions."""

    def test_empty_schema_returns_empty(self):
        """Empty schema should return empty constraints."""
        result = extract_type_level_constraints({})

        assert result["node_constraints"] == []
        assert result["edge_constraints"] == []

    def test_extract_node_constraint_from_dict(self):
        """Should extract node constraint from dict schema."""
        schema = {
            "node_types": {
                "TacticDef": {
                    "name": "TacticDef",
                    "properties": {},
                    "constraint": {
                        "create": "never",
                        "search": {"properties": [{"name": "id", "mode": "exact"}]}
                    }
                }
            }
        }

        result = extract_type_level_constraints(schema)

        assert len(result["node_constraints"]) == 1
        constraint = result["node_constraints"][0]
        assert constraint["node_type"] == "TacticDef"
        assert constraint["create"] == "never"

    def test_extract_edge_constraint_from_dict(self):
        """Should extract edge constraint from dict schema."""
        schema = {
            "relationship_types": {
                "MITIGATES": {
                    "name": "MITIGATES",
                    "allowed_source_types": ["SecurityBehavior"],
                    "allowed_target_types": ["TacticDef"],
                    "constraint": {
                        "create": "never"
                    }
                }
            }
        }

        result = extract_type_level_constraints(schema)

        assert len(result["edge_constraints"]) == 1
        constraint = result["edge_constraints"][0]
        assert constraint["edge_type"] == "MITIGATES"
        assert constraint["create"] == "never"
        # Should infer source_type and target_type from allowed types
        assert constraint.get("source_type") == "SecurityBehavior"
        assert constraint.get("target_type") == "TacticDef"

    def test_no_constraint_in_node_type(self):
        """Node type without constraint should not add to constraints."""
        schema = {
            "node_types": {
                "CallerTactic": {
                    "name": "CallerTactic",
                    "properties": {}
                    # No constraint field
                }
            }
        }

        result = extract_type_level_constraints(schema)

        assert len(result["node_constraints"]) == 0

    def test_multiple_allowed_types_no_inference(self):
        """Multiple allowed source/target types should not infer single type."""
        schema = {
            "relationship_types": {
                "MENTIONS": {
                    "name": "MENTIONS",
                    "allowed_source_types": ["Memory", "Conversation"],  # Multiple
                    "allowed_target_types": ["Person", "Task"],  # Multiple
                    "constraint": {
                        "create": "auto"
                    }
                }
            }
        }

        result = extract_type_level_constraints(schema)

        assert len(result["edge_constraints"]) == 1
        constraint = result["edge_constraints"][0]
        # Should NOT infer source/target since there are multiple options
        assert "source_type" not in constraint or constraint.get("source_type") is None
        assert "target_type" not in constraint or constraint.get("target_type") is None


class TestExtractOmoFieldsFromPolicy:
    """Test extraction of OMO safety fields."""

    def test_extract_defaults(self):
        """Should return defaults for empty policy."""
        result = extract_omo_fields_from_policy({})

        assert result["consent"] == DEFAULT_CONSENT
        assert result["risk"] == DEFAULT_RISK
        assert result["acl"] is None

    def test_extract_custom_values(self):
        """Should extract custom OMO values."""
        policy = {
            "consent": "explicit",
            "risk": "high",
            "acl": {"read": ["user1"]}
        }

        result = extract_omo_fields_from_policy(policy)

        assert result["consent"] == "explicit"
        assert result["risk"] == "high"
        assert result["acl"] == {"read": ["user1"]}


class TestShouldSkipGraphExtraction:
    """Test graph extraction skip logic."""

    def test_consent_none_skips_extraction(self):
        """consent='none' should skip graph extraction."""
        result = should_skip_graph_extraction({"consent": "none"})
        assert result is True

    def test_consent_implicit_does_not_skip(self):
        """consent='implicit' should not skip graph extraction."""
        result = should_skip_graph_extraction({"consent": "implicit"})
        assert result is False

    def test_consent_explicit_does_not_skip(self):
        """consent='explicit' should not skip graph extraction."""
        result = should_skip_graph_extraction({"consent": "explicit"})
        assert result is False

    def test_empty_policy_does_not_skip(self):
        """Empty policy should not skip graph extraction."""
        result = should_skip_graph_extraction({})
        assert result is False


class TestDeepTrustScenario:
    """Test DeepTrust security schema policy scenarios."""

    def test_schema_controlled_vocabulary(self):
        """DeepTrust schema should define controlled vocabulary constraints."""
        schema_policy = {
            "mode": "auto",
            "node_constraints": [
                {"node_type": "TacticDef", "create": "never"},
                {"node_type": "SecurityBehavior", "create": "never"},
                {"node_type": "Impact", "create": "never"}
            ]
        }

        result = merge_memory_policies(schema_policy, None)

        # All controlled vocabulary types should have create='never'
        for constraint in result["node_constraints"]:
            assert constraint["create"] == "never"

    def test_memory_relaxes_constraint(self):
        """Memory policy can relax schema constraint (if allowed)."""
        schema_policy = {
            "node_constraints": [
                {"node_type": "TacticDef", "create": "never"}
            ]
        }
        memory_policy = {
            "node_constraints": [
                {"node_type": "TacticDef", "create": "auto"}  # Override
            ]
        }

        result = merge_memory_policies(schema_policy, memory_policy)

        tactic_constraint = result["node_constraints"][0]
        assert tactic_constraint["create"] == "auto"  # Relaxed

    def test_memory_adds_dynamic_entity_constraint(self):
        """Memory can add constraints for dynamic entities."""
        schema_policy = {
            "node_constraints": [
                {"node_type": "TacticDef", "create": "never"}
            ]
        }
        memory_policy = {
            "node_constraints": [
                {
                    "node_type": "CallerTactic",
                    "create": "auto",
                    "when": {"severity": "critical"},
                    "set": {"alert_security_team": True}
                }
            ]
        }

        result = merge_memory_policies(schema_policy, memory_policy)

        assert len(result["node_constraints"]) == 2

        caller_tactic = next(c for c in result["node_constraints"] if c["node_type"] == "CallerTactic")
        assert caller_tactic["create"] == "auto"
        assert caller_tactic["when"] == {"severity": "critical"}
        assert caller_tactic["set"] == {"alert_security_team": True}

    def test_full_deeptrust_policy_merge(self, deeptrust_schema_dict):
        """Full DeepTrust schema policy merge scenario."""
        schema_policy = deeptrust_schema_dict.get("memory_policy", {})

        # Memory-level override for specific call
        memory_policy = {
            "node_constraints": [
                {
                    "node_type": "Violation",
                    "create": "auto",
                    "set": {
                        "behavior_id": "SB080",
                        "severity": {"mode": "auto"}
                    }
                }
            ],
            "edge_constraints": [
                {
                    "edge_type": "HAS_VIOLATION",
                    "source_type": "Conversation",
                    "target_type": "Violation",
                    "create": "auto"
                }
            ]
        }

        result = merge_memory_policies(schema_policy, memory_policy)

        assert result["mode"] == "auto"

        # Check violation constraint is present
        violation_constraint = next(
            (c for c in result["node_constraints"] if c["node_type"] == "Violation"),
            None
        )
        assert violation_constraint is not None
        assert violation_constraint["set"]["behavior_id"] == "SB080"


class TestDocumentedExamples:
    """Test examples from DX documentation."""

    def test_link_to_expansion_simple(self):
        """Test that simple link_to expands correctly conceptually."""
        # link_to="Task:title" expands to:
        # memory_policy=MemoryPolicy(
        #     node_constraints=[
        #         NodeConstraint(
        #             node_type="Task",
        #             search=SearchConfig(properties=[PropertyMatch(name="title", mode="semantic")])
        #         )
        #     ]
        # )

        memory_policy = {
            "node_constraints": [
                {
                    "node_type": "Task",
                    "search": {
                        "properties": [
                            {"name": "title", "mode": "semantic"}
                        ]
                    }
                }
            ]
        }

        result = merge_memory_policies(None, memory_policy)

        assert len(result["node_constraints"]) == 1
        assert result["node_constraints"][0]["node_type"] == "Task"

    def test_link_to_expansion_with_set(self):
        """Test link_to with set expands correctly."""
        # link_to={"Task:title": {"set": {"status": "completed"}}} expands to:
        memory_policy = {
            "node_constraints": [
                {
                    "node_type": "Task",
                    "search": {
                        "properties": [
                            {"name": "title", "mode": "semantic"}
                        ]
                    },
                    "set": {"status": "completed"}
                }
            ]
        }

        result = merge_memory_policies(None, memory_policy)

        assert result["node_constraints"][0]["set"]["status"] == "completed"

    def test_link_to_expansion_with_when(self):
        """Test link_to with when expands correctly."""
        # link_to={"Task:title": {"when": {"priority": "high"}}} expands to:
        memory_policy = {
            "node_constraints": [
                {
                    "node_type": "Task",
                    "search": {
                        "properties": [
                            {"name": "title", "mode": "semantic"}
                        ]
                    },
                    "when": {"priority": "high"}
                }
            ]
        }

        result = merge_memory_policies(None, memory_policy)

        assert result["node_constraints"][0]["when"]["priority"] == "high"

    def test_link_to_expansion_with_create(self):
        """Test link_to with create expands correctly."""
        # link_to={"Task:title": {"create": "never"}} expands to:
        memory_policy = {
            "node_constraints": [
                {
                    "node_type": "Task",
                    "search": {
                        "properties": [
                            {"name": "title", "mode": "semantic"}
                        ]
                    },
                    "create": "never"
                }
            ]
        }

        result = merge_memory_policies(None, memory_policy)

        assert result["node_constraints"][0]["create"] == "never"
