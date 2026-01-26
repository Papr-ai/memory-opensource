"""
Integration tests for memory policy system.

Tests the full pipeline flow:
- Schema-level constraints applied
- Memory-level overrides
- Edge constraints in pipeline
- Via relationship search
- Full DeepTrust E2E scenarios
- Set values with various modes

Run with: pytest tests/test_memory_policy/test_memory_policy_integration.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from services.memory_policy_resolver import (
    merge_memory_policies,
    extract_type_level_constraints,
    resolve_memory_policy_from_schema
)
from services.node_constraint_resolver import (
    apply_node_constraints,
    _evaluate_when_condition
)
from services.edge_constraint_resolver import apply_edge_constraints


class TestSchemaConstraintIntegration:
    """Test schema-level constraints flowing through the pipeline."""

    def test_schema_to_merged_policy_flow(self, deeptrust_schema_dict):
        """Schema constraints should flow into merged policy correctly."""
        # Extract type-level constraints from schema
        type_constraints = extract_type_level_constraints(deeptrust_schema_dict)

        # Verify controlled vocabulary types have create='never'
        tactic_constraint = next(
            (c for c in type_constraints["node_constraints"] if c.get("node_type") == "TacticDef"),
            None
        )
        assert tactic_constraint is not None
        assert tactic_constraint["create"] == "never"

        security_behavior_constraint = next(
            (c for c in type_constraints["node_constraints"] if c.get("node_type") == "SecurityBehavior"),
            None
        )
        assert security_behavior_constraint is not None
        assert security_behavior_constraint["create"] == "never"

    def test_dynamic_entities_allow_creation(self, deeptrust_schema_dict):
        """Dynamic entities should allow auto creation."""
        type_constraints = extract_type_level_constraints(deeptrust_schema_dict)

        caller_tactic = next(
            (c for c in type_constraints["node_constraints"] if c.get("node_type") == "CallerTactic"),
            None
        )
        assert caller_tactic is not None
        assert caller_tactic["create"] == "auto"

        conversation = next(
            (c for c in type_constraints["node_constraints"] if c.get("node_type") == "Conversation"),
            None
        )
        assert conversation is not None
        assert conversation["create"] == "auto"

    @pytest.mark.asyncio
    async def test_controlled_vocabulary_enforced(self, mock_memory_graph, deeptrust_schema_dict):
        """Controlled vocabulary should prevent creation of unknown nodes."""
        # Extract constraints from schema
        type_constraints = extract_type_level_constraints(deeptrust_schema_dict)

        # Attempt to create unknown TacticDef
        node = {"type": "TacticDef", "properties": {"id": "TA9999"}}
        extracted = {"id": "TA9999", "name": "Unknown Tactic"}

        # Mock returns no existing node
        mock_memory_graph.find_node_by_property = AsyncMock(return_value=None)
        mock_memory_graph.find_node_by_semantic_match = AsyncMock(return_value=None)

        should_create, existing, props = await apply_node_constraints(
            node=node,
            node_type="TacticDef",
            node_constraints=type_constraints["node_constraints"],
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        # create='never' should block creation
        assert should_create is False
        assert existing is None


class TestMemoryOverrideIntegration:
    """Test memory-level overrides of schema policies."""

    def test_memory_relaxes_schema_constraint(self, deeptrust_schema_dict):
        """Memory policy can relax schema's create='never' to 'auto'."""
        type_constraints = extract_type_level_constraints(deeptrust_schema_dict)

        # Memory override relaxes TacticDef constraint
        memory_policy = {
            "node_constraints": [
                {"node_type": "TacticDef", "create": "auto"}  # Override
            ]
        }

        # Build schema policy from type constraints
        schema_policy = {
            "node_constraints": type_constraints["node_constraints"],
            "edge_constraints": type_constraints["edge_constraints"]
        }

        merged = merge_memory_policies(schema_policy, memory_policy)

        # Find TacticDef constraint in merged
        tactic_constraint = next(
            (c for c in merged["node_constraints"] if c.get("node_type") == "TacticDef"),
            None
        )
        assert tactic_constraint is not None
        assert tactic_constraint["create"] == "auto"  # Memory override won

    def test_memory_adds_conditional_constraint(self, deeptrust_schema_dict):
        """Memory can add conditional constraints for specific processing."""
        type_constraints = extract_type_level_constraints(deeptrust_schema_dict)

        # Memory adds conditional alert for critical violations
        memory_policy = {
            "node_constraints": [
                {
                    "node_type": "Violation",
                    "when": {"severity": "critical"},
                    "set": {"requires_immediate_review": True, "escalated": True}
                }
            ]
        }

        schema_policy = {"node_constraints": type_constraints["node_constraints"]}
        merged = merge_memory_policies(schema_policy, memory_policy)

        # Find Violation constraint
        violation_constraints = [
            c for c in merged["node_constraints"]
            if c.get("node_type") == "Violation"
        ]

        # Should have the conditional constraint
        conditional = next(
            (c for c in violation_constraints if c.get("when")),
            None
        )
        assert conditional is not None
        assert conditional["when"] == {"severity": "critical"}
        assert conditional["set"]["escalated"] is True

    @pytest.mark.asyncio
    async def test_memory_override_enables_creation(self, mock_memory_graph, deeptrust_schema_dict):
        """Memory override should enable creation when schema prohibits."""
        # First test: Schema constraint blocks
        schema_constraints = extract_type_level_constraints(deeptrust_schema_dict)["node_constraints"]

        node = {"type": "TacticDef", "properties": {}}
        extracted = {"id": "NEW001", "name": "New Tactic"}

        mock_memory_graph.find_node_by_property = AsyncMock(return_value=None)
        mock_memory_graph.find_node_by_semantic_match = AsyncMock(return_value=None)

        # Schema says create='never'
        should_create, _, _ = await apply_node_constraints(
            node=node,
            node_type="TacticDef",
            node_constraints=schema_constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )
        assert should_create is False

        # Now with memory override
        memory_constraints = [
            {"node_type": "TacticDef", "create": "auto"}
        ]
        merged_constraints = merge_memory_policies(
            {"node_constraints": schema_constraints},
            {"node_constraints": memory_constraints}
        )["node_constraints"]

        should_create, _, _ = await apply_node_constraints(
            node=node,
            node_type="TacticDef",
            node_constraints=merged_constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )
        assert should_create is True  # Memory override enabled creation


class TestEdgeConstraintIntegration:
    """Test edge constraints in the full pipeline."""

    @pytest.mark.asyncio
    async def test_edge_controlled_vocabulary_requires_existing(
        self,
        mock_memory_graph,
        deeptrust_schema_dict
    ):
        """Edge to controlled vocabulary should require existing target."""
        edge_constraints = extract_type_level_constraints(deeptrust_schema_dict)["edge_constraints"]

        source_node = {"type": "SecurityBehavior", "properties": {"id": "SB080"}}
        target_node = {"type": "TacticDef", "properties": {"id": "TA9999"}}  # Unknown

        mock_memory_graph.find_node_by_property = AsyncMock(return_value=None)
        mock_memory_graph.find_node_by_semantic_match = AsyncMock(return_value=None)

        should_create, final_target, props = await apply_edge_constraints(
            source_node=source_node,
            target_node=target_node,
            edge_type="MITIGATES",
            edge_constraints=edge_constraints,
            memory_graph=mock_memory_graph,
            extracted_edge_properties={}
        )

        # MITIGATES has create='never', no existing target -> blocked
        assert should_create is False

    @pytest.mark.asyncio
    async def test_edge_finds_and_links_existing(
        self,
        mock_memory_graph,
        deeptrust_schema_dict
    ):
        """Edge should find existing target and link to it."""
        edge_constraints = extract_type_level_constraints(deeptrust_schema_dict)["edge_constraints"]

        source_node = {"type": "SecurityBehavior", "properties": {"id": "SB080"}}
        target_node = {"type": "TacticDef", "properties": {"id": "TA0005"}}

        # Return existing TacticDef
        mock_memory_graph.find_node_by_property = AsyncMock(return_value={
            "id": "existing_tactic",
            "type": "TacticDef",
            "properties": {"id": "TA0005", "name": "Defense Evasion"}
        })

        should_create, final_target, props = await apply_edge_constraints(
            source_node=source_node,
            target_node=target_node,
            edge_type="MITIGATES",
            edge_constraints=edge_constraints,
            memory_graph=mock_memory_graph,
            extracted_edge_properties={}
        )

        assert should_create is True
        assert final_target["properties"]["id"] == "TA0005"


class TestViaRelationshipIntegration:
    """Test via_relationship search in the pipeline."""

    @pytest.mark.asyncio
    async def test_via_relationship_finds_connected_node(self, mock_memory_graph):
        """Via relationship should find node through relationship."""
        # SecurityBehavior constraint with via_relationship
        constraints = [
            {
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
        ]

        # Node properties must contain search values for the search to work
        node = {"type": "SecurityBehavior", "properties": {"name": "Verify Identity", "trigger_context": "Defense Evasion"}}
        extracted = {"name": "Verify Identity", "trigger_context": "Defense Evasion"}

        # Setup mocks
        # Direct semantic search returns nothing on first call, then finds TacticDef on second
        mock_memory_graph.find_node_by_semantic_match = AsyncMock(side_effect=[
            None,  # Direct SecurityBehavior search - no match
            {"id": "tactic_1", "type": "TacticDef", "properties": {"name": "Defense Evasion"}}  # TacticDef found via target_search
        ])

        # Via relationship finds SecurityBehavior
        mock_memory_graph.find_node_via_relationship = AsyncMock(return_value={
            "id": "behavior_1",
            "type": "SecurityBehavior",
            "properties": {"id": "SB080", "name": "Verify Identity"}
        })

        should_create, existing, props = await apply_node_constraints(
            node=node,
            node_type="SecurityBehavior",
            node_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        # Should find existing via relationship
        assert should_create is False
        assert existing is not None


class TestDeepTrustScenario:
    """Full E2E tests with DeepTrust security domain."""

    @pytest.mark.asyncio
    async def test_security_behavior_mitigates_known_tactic(
        self,
        mock_memory_graph,
        sample_security_behavior_node,
        sample_tactic_def_node
    ):
        """SecurityBehavior should link to known TacticDef via MITIGATES."""
        # Schema constraints
        node_constraints = [
            {
                "node_type": "SecurityBehavior",
                "create": "never",
                "search": {"properties": [{"name": "id", "mode": "exact"}]}
            },
            {
                "node_type": "TacticDef",
                "create": "never",
                "search": {"properties": [{"name": "id", "mode": "exact"}]}
            }
        ]
        edge_constraints = [
            {
                "edge_type": "MITIGATES",
                "source_type": "SecurityBehavior",
                "target_type": "TacticDef",
                "create": "never",
                "search": {"properties": [{"name": "id", "mode": "exact"}]}
            }
        ]

        # Find existing SecurityBehavior
        mock_memory_graph.find_node_by_property = AsyncMock(return_value={
            "id": "sb_1",
            "type": "SecurityBehavior",
            "properties": sample_security_behavior_node["properties"]
        })

        sb_should_create, sb_existing, _ = await apply_node_constraints(
            node=sample_security_behavior_node,
            node_type="SecurityBehavior",
            node_constraints=node_constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=sample_security_behavior_node["properties"]
        )

        assert sb_should_create is False
        assert sb_existing is not None

        # Find existing TacticDef for edge
        mock_memory_graph.find_node_by_property = AsyncMock(return_value={
            "id": "tactic_1",
            "type": "TacticDef",
            "properties": sample_tactic_def_node["properties"]
        })

        edge_should_create, edge_target, _ = await apply_edge_constraints(
            source_node=sb_existing,
            target_node=sample_tactic_def_node,
            edge_type="MITIGATES",
            edge_constraints=edge_constraints,
            memory_graph=mock_memory_graph,
            extracted_edge_properties={}
        )

        assert edge_should_create is True
        assert edge_target["properties"]["id"] == "TA0005"

    @pytest.mark.asyncio
    async def test_caller_tactic_creates_and_links(
        self,
        mock_memory_graph,
        sample_caller_tactic_node
    ):
        """CallerTactic should create and link to existing TacticDef."""
        node_constraints = [
            {"node_type": "CallerTactic", "create": "auto"},
            {"node_type": "TacticDef", "create": "never"}
        ]
        edge_constraints = [
            {
                "edge_type": "IS_INSTANCE",
                "source_type": "CallerTactic",
                "target_type": "TacticDef",
                "create": "auto",
                "search": {"properties": [{"name": "id", "mode": "exact"}]}
            }
        ]

        # CallerTactic should be created
        mock_memory_graph.find_node_by_semantic_match = AsyncMock(return_value=None)

        ct_should_create, ct_existing, ct_props = await apply_node_constraints(
            node=sample_caller_tactic_node,
            node_type="CallerTactic",
            node_constraints=node_constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=sample_caller_tactic_node["properties"]
        )

        assert ct_should_create is True

        # IS_INSTANCE edge should find existing TacticDef
        tactic_node = {"type": "TacticDef", "properties": {"id": "TA0005"}}
        mock_memory_graph.find_node_by_property = AsyncMock(return_value={
            "id": "tactic_1",
            "type": "TacticDef",
            "properties": {"id": "TA0005", "name": "Defense Evasion"}
        })

        edge_should_create, edge_target, _ = await apply_edge_constraints(
            source_node=sample_caller_tactic_node,
            target_node=tactic_node,
            edge_type="IS_INSTANCE",
            edge_constraints=edge_constraints,
            memory_graph=mock_memory_graph,
            extracted_edge_properties={"confidence": 0.95}
        )

        assert edge_should_create is True
        assert edge_target["properties"]["id"] == "TA0005"

    @pytest.mark.asyncio
    async def test_conditional_violation_alert(self, mock_memory_graph, sample_violation_node):
        """Critical Violation should trigger alert set values."""
        node_constraints = [
            {
                "node_type": "Violation",
                "create": "auto",
                "when": {"severity": "critical"},
                "set": {
                    "requires_immediate_review": True,
                    "escalated": True,
                    "alert_security_team": True
                }
            }
        ]

        should_create, _, props = await apply_node_constraints(
            node=sample_violation_node,
            node_type="Violation",
            node_constraints=node_constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=sample_violation_node["properties"]
        )

        assert should_create is True
        assert props["requires_immediate_review"] is True
        assert props["escalated"] is True
        assert props["alert_security_team"] is True

    def test_when_clause_conditional_applies(self):
        """When clause should conditionally apply constraint."""
        # Critical severity matches
        critical_result = _evaluate_when_condition(
            condition={"severity": "critical"},
            node_properties={"severity": "critical", "behavior_id": "SB080"}
        )
        assert critical_result is True

        # Non-critical doesn't match
        non_critical_result = _evaluate_when_condition(
            condition={"severity": "critical"},
            node_properties={"severity": "high", "behavior_id": "SB080"}
        )
        assert non_critical_result is False


class TestSetValuesIntegration:
    """Test set clause with various modes in integration."""

    @pytest.mark.asyncio
    async def test_set_exact_value(self, mock_memory_graph):
        """Set with exact value should override extracted."""
        node_constraints = [
            {
                "node_type": "Violation",
                "create": "auto",
                "set": {
                    "workspace_id": "ws_security_001",
                    "source": "call_analyzer"
                }
            }
        ]
        extracted = {"behavior_id": "SB080", "severity": "critical"}

        _, _, props = await apply_node_constraints(
            node={"type": "Violation", "properties": {}},
            node_type="Violation",
            node_constraints=node_constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        assert props["workspace_id"] == "ws_security_001"  # Set exact
        assert props["source"] == "call_analyzer"  # Set exact
        assert props["behavior_id"] == "SB080"  # Preserved from extracted

    @pytest.mark.asyncio
    async def test_set_auto_mode_preserves_extracted(self, mock_memory_graph):
        """Set with mode='auto' should preserve extracted value."""
        node_constraints = [
            {
                "node_type": "Violation",
                "create": "auto",
                "set": {
                    "severity": {"mode": "auto"},
                    "workspace_id": "ws_001"
                }
            }
        ]
        extracted = {"behavior_id": "SB080", "severity": "critical"}

        _, _, props = await apply_node_constraints(
            node={"type": "Violation", "properties": {}},
            node_type="Violation",
            node_constraints=node_constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        assert props["severity"] == "critical"  # Auto preserved extracted
        assert props["workspace_id"] == "ws_001"  # Exact value

    @pytest.mark.asyncio
    async def test_mixed_set_modes(self, mock_memory_graph):
        """Test mix of exact values and auto mode."""
        node_constraints = [
            {
                "node_type": "Conversation",
                "create": "auto",
                "search": {"properties": [{"name": "call_id", "mode": "exact"}]},
                "set": {
                    "workspace_id": "ws_call_center",  # Exact
                    "risk_level": {"mode": "auto"},  # Auto extract
                    "security_score": {"mode": "auto"},  # Auto extract
                    "processed": True  # Exact
                }
            }
        ]
        extracted = {
            "call_id": "call_4492",
            "risk_level": "high",
            "security_score": 35,
            "duration": 300
        }

        mock_memory_graph.find_node_by_property = AsyncMock(return_value=None)

        _, _, props = await apply_node_constraints(
            node={"type": "Conversation", "properties": {}},
            node_type="Conversation",
            node_constraints=node_constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        assert props["workspace_id"] == "ws_call_center"  # Exact
        assert props["risk_level"] == "high"  # Auto preserved
        assert props["security_score"] == 35  # Auto preserved
        assert props["processed"] is True  # Exact
        assert props["duration"] == 300  # Preserved from extracted


class TestPolicyPrecedenceIntegration:
    """Test full policy precedence: System < Type-level < Schema < Memory."""

    @pytest.mark.asyncio
    async def test_full_precedence_chain(self, mock_memory_graph, deeptrust_schema_dict):
        """Test complete precedence chain."""
        # 1. Extract type-level constraints (from schema type definitions)
        type_constraints = extract_type_level_constraints(deeptrust_schema_dict)

        # 2. Schema-level memory_policy (in schema root)
        schema_memory_policy = deeptrust_schema_dict.get("memory_policy", {})

        # 3. Merge type-level with schema-level
        schema_policy = merge_memory_policies(
            {"node_constraints": type_constraints["node_constraints"]},
            schema_memory_policy
        )

        # Verify TacticDef still has create='never' from type-level
        tactic = next(
            (c for c in schema_policy["node_constraints"] if c.get("node_type") == "TacticDef"),
            None
        )
        assert tactic["create"] == "never"

        # 4. Memory-level override
        memory_policy = {
            "node_constraints": [
                {"node_type": "TacticDef", "create": "auto"}  # Override
            ]
        }

        final_policy = merge_memory_policies(schema_policy, memory_policy)

        # Memory level should win
        tactic = next(
            (c for c in final_policy["node_constraints"] if c.get("node_type") == "TacticDef"),
            None
        )
        assert tactic["create"] == "auto"  # Memory override won

    def test_mode_precedence(self):
        """Mode should follow precedence chain."""
        schema_policy = {"mode": "structured"}
        memory_policy = {"mode": "auto"}

        result = merge_memory_policies(schema_policy, memory_policy)
        assert result["mode"] == "auto"

    def test_consent_precedence(self):
        """Consent should follow precedence chain."""
        schema_policy = {"consent": "implicit"}
        memory_policy = {"consent": "explicit"}

        result = merge_memory_policies(schema_policy, memory_policy)
        assert result["consent"] == "explicit"


class TestResolveMemoryPolicyFromSchema:
    """Test the full resolve_memory_policy_from_schema function."""

    @pytest.mark.asyncio
    async def test_resolve_with_schema_id(self, mock_memory_graph, deeptrust_schema_dict):
        """Should resolve policy from schema when schema_id provided."""
        # Mock schema retrieval
        mock_memory_graph.get_user_schema_async = AsyncMock(return_value=deeptrust_schema_dict)

        result = await resolve_memory_policy_from_schema(
            memory_graph=mock_memory_graph,
            schema_id="deeptrust_security",
            memory_policy=None,
            user_id="test_user"
        )

        # Should have constraints from schema
        assert len(result["node_constraints"]) > 0
        assert result["mode"] == "auto"

    @pytest.mark.asyncio
    async def test_resolve_with_memory_override(self, mock_memory_graph, deeptrust_schema_dict):
        """Should apply memory override on top of schema."""
        mock_memory_graph.get_user_schema_async = AsyncMock(return_value=deeptrust_schema_dict)

        memory_policy = {
            "node_constraints": [
                {"node_type": "TacticDef", "create": "auto"}
            ]
        }

        result = await resolve_memory_policy_from_schema(
            memory_graph=mock_memory_graph,
            schema_id="deeptrust_security",
            memory_policy=memory_policy,
            user_id="test_user"
        )

        # Memory override should be applied
        tactic = next(
            (c for c in result["node_constraints"] if c.get("node_type") == "TacticDef"),
            None
        )
        assert tactic["create"] == "auto"

    @pytest.mark.asyncio
    async def test_resolve_without_schema_id(self, mock_memory_graph):
        """Should use defaults and memory policy when no schema_id."""
        memory_policy = {
            "mode": "auto",
            "node_constraints": [
                {"node_type": "Task", "create": "auto"}
            ]
        }

        result = await resolve_memory_policy_from_schema(
            memory_graph=mock_memory_graph,
            schema_id=None,
            memory_policy=memory_policy
        )

        assert result["mode"] == "auto"
        assert len(result["node_constraints"]) == 1

    @pytest.mark.asyncio
    async def test_resolve_handles_schema_fetch_error(self, mock_memory_graph):
        """Should handle schema fetch errors gracefully."""
        mock_memory_graph.get_user_schema_async = AsyncMock(side_effect=Exception("Schema not found"))

        memory_policy = {"mode": "auto"}

        # Should not raise, should use defaults + memory policy
        result = await resolve_memory_policy_from_schema(
            memory_graph=mock_memory_graph,
            schema_id="nonexistent_schema",
            memory_policy=memory_policy
        )

        assert result["mode"] == "auto"
