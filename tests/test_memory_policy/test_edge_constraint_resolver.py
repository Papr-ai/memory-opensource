"""
Tests for edge constraint resolver - applying edge constraints during graph generation.

Tests the apply_edge_constraints function which handles:
- Finding applicable constraints by edge_type, source_type, target_type
- Evaluating 'when' conditions for edges
- Searching for existing target nodes
- Applying create policy (auto vs never)
- Setting edge property values

Run with: pytest tests/test_memory_policy/test_edge_constraint_resolver.py -v
"""

import pytest
from unittest.mock import AsyncMock
from services.edge_constraint_resolver import (
    apply_edge_constraints,
    _find_applicable_constraint,
    _apply_set_values,
    get_edge_constraints_for_type,
    validate_edge_constraints
)


class TestFindApplicableEdgeConstraint:
    """Test finding the most applicable edge constraint."""

    def test_exact_edge_type_match(self):
        """Should find constraint with exact edge_type match."""
        constraints = [
            {"edge_type": "MITIGATES", "create": "never"},
            {"edge_type": "IS_INSTANCE", "create": "auto"}
        ]

        result = _find_applicable_constraint(
            edge_type="MITIGATES",
            source_type="SecurityBehavior",
            target_type="TacticDef",
            edge_constraints=constraints
        )

        assert result is not None
        assert result["edge_type"] == "MITIGATES"

    def test_no_matching_constraint(self):
        """Should return None when no constraint matches."""
        constraints = [
            {"edge_type": "MITIGATES", "create": "never"}
        ]

        result = _find_applicable_constraint(
            edge_type="IS_INSTANCE",
            source_type="CallerTactic",
            target_type="TacticDef",
            edge_constraints=constraints
        )

        assert result is None

    def test_match_with_source_type(self):
        """Should match constraint with source_type filter."""
        constraints = [
            {"edge_type": "MITIGATES", "source_type": "SecurityBehavior", "create": "never"},
            {"edge_type": "MITIGATES", "source_type": "Policy", "create": "auto"}
        ]

        result = _find_applicable_constraint(
            edge_type="MITIGATES",
            source_type="SecurityBehavior",
            target_type="TacticDef",
            edge_constraints=constraints
        )

        assert result is not None
        assert result["source_type"] == "SecurityBehavior"
        assert result["create"] == "never"

    def test_source_type_mismatch_no_match(self):
        """Should not match when source_type doesn't match."""
        constraints = [
            {"edge_type": "MITIGATES", "source_type": "Policy", "create": "never"}
        ]

        result = _find_applicable_constraint(
            edge_type="MITIGATES",
            source_type="SecurityBehavior",
            target_type="TacticDef",
            edge_constraints=constraints
        )

        assert result is None

    def test_match_with_target_type(self):
        """Should match constraint with target_type filter."""
        constraints = [
            {"edge_type": "MITIGATES", "target_type": "TacticDef", "create": "never"},
            {"edge_type": "MITIGATES", "target_type": "Impact", "create": "auto"}
        ]

        result = _find_applicable_constraint(
            edge_type="MITIGATES",
            source_type="SecurityBehavior",
            target_type="TacticDef",
            edge_constraints=constraints
        )

        assert result is not None
        assert result["target_type"] == "TacticDef"

    def test_target_type_mismatch_no_match(self):
        """Should not match when target_type doesn't match."""
        constraints = [
            {"edge_type": "MITIGATES", "target_type": "Impact", "create": "never"}
        ]

        result = _find_applicable_constraint(
            edge_type="MITIGATES",
            source_type="SecurityBehavior",
            target_type="TacticDef",
            edge_constraints=constraints
        )

        assert result is None

    def test_full_composite_key_match(self):
        """Should match constraint with full composite key."""
        constraints = [
            {
                "edge_type": "MITIGATES",
                "source_type": "SecurityBehavior",
                "target_type": "TacticDef",
                "create": "never"
            }
        ]

        result = _find_applicable_constraint(
            edge_type="MITIGATES",
            source_type="SecurityBehavior",
            target_type="TacticDef",
            edge_constraints=constraints
        )

        assert result is not None
        assert result["create"] == "never"

    def test_specificity_priority(self):
        """More specific constraint should be prioritized."""
        constraints = [
            {"edge_type": "MITIGATES", "create": "auto"},  # Less specific
            {
                "edge_type": "MITIGATES",
                "source_type": "SecurityBehavior",
                "target_type": "TacticDef",
                "create": "never"
            }  # More specific
        ]

        result = _find_applicable_constraint(
            edge_type="MITIGATES",
            source_type="SecurityBehavior",
            target_type="TacticDef",
            edge_constraints=constraints
        )

        assert result is not None
        assert result["create"] == "never"  # More specific wins

    def test_empty_constraints(self):
        """Should return None for empty constraints."""
        result = _find_applicable_constraint(
            edge_type="MITIGATES",
            source_type="SecurityBehavior",
            target_type="TacticDef",
            edge_constraints=[]
        )

        assert result is None


class TestApplyEdgeConstraints:
    """Test the full edge constraint application flow."""

    @pytest.mark.asyncio
    async def test_no_constraints_allows_creation(self, mock_memory_graph):
        """No constraints should allow edge creation."""
        source_node = {"type": "SecurityBehavior", "properties": {"id": "SB080"}}
        target_node = {"type": "TacticDef", "properties": {"id": "TA0005"}}
        extracted = {"confidence": 0.95}

        should_create, final_target, props = await apply_edge_constraints(
            source_node=source_node,
            target_node=target_node,
            edge_type="MITIGATES",
            edge_constraints=[],
            memory_graph=mock_memory_graph,
            extracted_edge_properties=extracted
        )

        assert should_create is True
        assert final_target == target_node
        assert props == extracted

    @pytest.mark.asyncio
    async def test_no_matching_constraint_allows_creation(self, mock_memory_graph):
        """No matching constraint should allow edge creation."""
        source_node = {"type": "CallerTactic", "properties": {}}
        target_node = {"type": "TacticDef", "properties": {}}
        constraints = [
            {"edge_type": "MITIGATES", "create": "never"}  # Different type
        ]
        extracted = {"confidence": 0.9}

        should_create, final_target, props = await apply_edge_constraints(
            source_node=source_node,
            target_node=target_node,
            edge_type="IS_INSTANCE",
            edge_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_edge_properties=extracted
        )

        assert should_create is True
        assert final_target == target_node

    @pytest.mark.asyncio
    async def test_create_never_no_existing_blocks_edge(self, mock_memory_graph):
        """create='never' with no existing target should block edge creation."""
        source_node = {"type": "SecurityBehavior", "properties": {"id": "SB080"}}
        target_node = {"type": "TacticDef", "properties": {"id": "TA9999"}}
        constraints = [
            {
                "edge_type": "MITIGATES",
                "create": "never",
                "search": {"properties": [{"name": "id", "mode": "exact"}]}
            }
        ]

        mock_memory_graph.find_node_by_property = AsyncMock(return_value=None)

        should_create, final_target, props = await apply_edge_constraints(
            source_node=source_node,
            target_node=target_node,
            edge_type="MITIGATES",
            edge_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_edge_properties={}
        )

        assert should_create is False
        assert final_target is None
        assert props is None

    @pytest.mark.asyncio
    async def test_create_never_with_existing_allows_edge(self, mock_memory_graph):
        """create='never' with existing target should allow edge."""
        source_node = {"type": "SecurityBehavior", "properties": {"id": "SB080"}}
        target_node = {"type": "TacticDef", "properties": {"id": "TA0005"}}
        constraints = [
            {
                "edge_type": "MITIGATES",
                "create": "never",
                "search": {"properties": [{"name": "id", "mode": "exact"}]}
            }
        ]

        # Mock returns existing target
        mock_memory_graph.find_node_by_property = AsyncMock(return_value={
            "id": "existing_tactic",
            "type": "TacticDef",
            "properties": {"id": "TA0005", "name": "Defense Evasion"}
        })

        should_create, final_target, props = await apply_edge_constraints(
            source_node=source_node,
            target_node=target_node,
            edge_type="MITIGATES",
            edge_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_edge_properties={"confidence": 0.95}
        )

        assert should_create is True
        assert final_target is not None
        assert final_target["properties"]["id"] == "TA0005"

    @pytest.mark.asyncio
    async def test_create_auto_allows_edge(self, mock_memory_graph):
        """create='auto' should allow edge creation."""
        source_node = {"type": "CallerTactic", "properties": {}}
        target_node = {"type": "TacticDef", "properties": {"id": "TA0005"}}
        constraints = [
            {
                "edge_type": "IS_INSTANCE",
                "create": "auto"
            }
        ]

        should_create, final_target, props = await apply_edge_constraints(
            source_node=source_node,
            target_node=target_node,
            edge_type="IS_INSTANCE",
            edge_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_edge_properties={}
        )

        assert should_create is True
        assert final_target == target_node

    @pytest.mark.asyncio
    async def test_when_condition_blocks_constraint(self, mock_memory_graph):
        """When condition not met should skip constraint."""
        source_node = {"type": "SecurityBehavior", "properties": {}}
        target_node = {"type": "TacticDef", "properties": {}}
        constraints = [
            {
                "edge_type": "MITIGATES",
                "create": "never",
                "when": {"severity": "critical"}  # Condition
            }
        ]
        extracted = {"severity": "low"}  # Not critical

        should_create, final_target, props = await apply_edge_constraints(
            source_node=source_node,
            target_node=target_node,
            edge_type="MITIGATES",
            edge_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_edge_properties=extracted
        )

        # When not met, constraint skipped, creation allowed
        assert should_create is True
        assert final_target == target_node

    @pytest.mark.asyncio
    async def test_when_condition_applies_constraint(self, mock_memory_graph):
        """When condition met should apply constraint."""
        source_node = {"type": "SecurityBehavior", "properties": {}}
        target_node = {"type": "TacticDef", "properties": {"id": "TA0005"}}
        constraints = [
            {
                "edge_type": "MITIGATES",
                "create": "never",
                "when": {"severity": "critical"}
            }
        ]
        extracted = {"severity": "critical"}

        mock_memory_graph.find_node_by_property = AsyncMock(return_value=None)

        should_create, final_target, props = await apply_edge_constraints(
            source_node=source_node,
            target_node=target_node,
            edge_type="MITIGATES",
            edge_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_edge_properties=extracted
        )

        # When met, create='never' blocks without existing target
        assert should_create is False


class TestApplyEdgeSetValues:
    """Test edge property value application from 'set' clause."""

    def test_exact_value_override(self):
        """Exact values should override extracted values."""
        constraint = {
            "set": {"confidence": 1.0, "verified": True}
        }
        extracted = {"confidence": 0.85}

        result = _apply_set_values(constraint, extracted)

        assert result["confidence"] == 1.0  # Overridden
        assert result["verified"] is True  # Added

    def test_auto_mode_preserves_extracted(self):
        """mode='auto' should preserve extracted value."""
        constraint = {
            "set": {"confidence": {"mode": "auto"}}
        }
        extracted = {"confidence": 0.95}

        result = _apply_set_values(constraint, extracted)

        assert result["confidence"] == 0.95  # Preserved

    def test_no_set_returns_extracted(self):
        """No set values should return extracted unchanged."""
        constraint = {}
        extracted = {"confidence": 0.9}

        result = _apply_set_values(constraint, extracted)

        assert result == extracted


class TestEdgeTextModeSetValues:
    """Test text_mode options for edge properties: replace, append, merge."""

    def test_edge_text_mode_replace_default(self):
        """text_mode='replace' (default) should overwrite existing value."""
        constraint = {
            "set": {"description": {"mode": "auto"}}  # Default text_mode is 'replace'
        }
        extracted = {"description": "New relationship description"}
        existing = {"description": "Old relationship description"}

        result = _apply_set_values(constraint, extracted, existing)

        assert result["description"] == "New relationship description"

    def test_edge_text_mode_append_string(self):
        """text_mode='append' should add to end of existing string."""
        constraint = {
            "set": {"audit_log": {"mode": "auto", "text_mode": "append"}}
        }
        extracted = {"audit_log": "2026-01-25: Relationship verified"}
        existing = {"audit_log": "2026-01-20: Relationship created"}

        result = _apply_set_values(constraint, extracted, existing)

        assert result["audit_log"] == "2026-01-20: Relationship created\n2026-01-25: Relationship verified"

    def test_edge_text_mode_append_list(self):
        """text_mode='append' should add to existing list."""
        constraint = {
            "set": {"events": {"mode": "auto", "text_mode": "append"}}
        }
        extracted = {"events": "verification_passed"}
        existing = {"events": ["created", "updated"]}

        result = _apply_set_values(constraint, extracted, existing)

        assert result["events"] == ["created", "updated", "verification_passed"]

    def test_edge_text_mode_merge(self):
        """text_mode='merge' should mark for intelligent LLM merge."""
        constraint = {
            "set": {"rationale": {"mode": "auto", "text_mode": "merge"}}
        }
        extracted = {"rationale": "Verified by security team review"}
        existing = {"rationale": "Initial matching based on semantic similarity"}

        result = _apply_set_values(constraint, extracted, existing)

        assert result["rationale"]["_merge"] is True
        assert result["rationale"]["existing"] == "Initial matching based on semantic similarity"
        assert result["rationale"]["new"] == "Verified by security team review"

    def test_edge_mixed_text_modes(self):
        """Multiple edge properties with different text_modes should work."""
        constraint = {
            "set": {
                "confidence": {"mode": "auto"},  # replace (default)
                "audit_log": {"mode": "auto", "text_mode": "append"},
                "effectiveness": "high"  # exact value
            }
        }
        extracted = {
            "confidence": 0.98,
            "audit_log": "Updated after verification"
        }
        existing = {
            "confidence": 0.85,
            "audit_log": "Initial log entry"
        }

        result = _apply_set_values(constraint, extracted, existing)

        # Replace (default)
        assert result["confidence"] == 0.98
        # Append
        assert result["audit_log"] == "Initial log entry\nUpdated after verification"
        # Exact
        assert result["effectiveness"] == "high"

    def test_edge_text_mode_append_no_existing(self):
        """text_mode='append' with no existing value should use extracted as-is."""
        constraint = {
            "set": {"notes": {"mode": "auto", "text_mode": "append"}}
        }
        extracted = {"notes": "First note"}
        existing = {}

        result = _apply_set_values(constraint, extracted, existing)

        assert result["notes"] == "First note"

    def test_edge_text_mode_no_extracted_keeps_existing(self):
        """text_mode='append' with no extracted value should keep existing."""
        constraint = {
            "set": {"notes": {"mode": "auto", "text_mode": "append"}}
        }
        extracted = {}
        existing = {"notes": "Original notes"}

        result = _apply_set_values(constraint, extracted, existing)

        assert result["notes"] == "Original notes"


class TestGetEdgeConstraintsForType:
    """Test getting all edge constraints for an edge type."""

    def test_exact_edge_type_match(self):
        """Should return constraints matching exact edge type."""
        constraints = [
            {"edge_type": "MITIGATES", "create": "never"},
            {"edge_type": "IS_INSTANCE", "create": "auto"},
            {"edge_type": "MITIGATES", "source_type": "Policy", "create": "auto"}
        ]

        result = get_edge_constraints_for_type("MITIGATES", constraints)

        assert len(result) == 2
        for c in result:
            assert c["edge_type"] == "MITIGATES"

    def test_with_source_type_filter(self):
        """Should filter by source_type when provided."""
        constraints = [
            {"edge_type": "MITIGATES", "source_type": "SecurityBehavior", "create": "never"},
            {"edge_type": "MITIGATES", "source_type": "Policy", "create": "auto"}
        ]

        result = get_edge_constraints_for_type(
            "MITIGATES",
            constraints,
            source_type="SecurityBehavior"
        )

        assert len(result) == 1
        assert result[0]["source_type"] == "SecurityBehavior"

    def test_with_target_type_filter(self):
        """Should filter by target_type when provided."""
        constraints = [
            {"edge_type": "MITIGATES", "target_type": "TacticDef", "create": "never"},
            {"edge_type": "MITIGATES", "target_type": "Impact", "create": "auto"}
        ]

        result = get_edge_constraints_for_type(
            "MITIGATES",
            constraints,
            target_type="TacticDef"
        )

        assert len(result) == 1
        assert result[0]["target_type"] == "TacticDef"

    def test_no_matches_returns_empty(self):
        """Should return empty list when no matches."""
        constraints = [
            {"edge_type": "MITIGATES", "create": "never"}
        ]

        result = get_edge_constraints_for_type("IS_INSTANCE", constraints)

        assert result == []


class TestValidateEdgeConstraints:
    """Test edge constraint validation."""

    def test_valid_constraint(self):
        """Valid constraint should pass validation."""
        constraints = [
            {
                "edge_type": "MITIGATES",
                "source_type": "SecurityBehavior",
                "target_type": "TacticDef",
                "create": "never"
            }
        ]

        errors = validate_edge_constraints(constraints)

        assert errors == []

    def test_missing_edge_type(self):
        """Missing edge_type should produce error."""
        constraints = [
            {"create": "never"}  # No edge_type
        ]

        errors = validate_edge_constraints(constraints)

        assert len(errors) == 1
        assert "edge_type is required" in errors[0]

    def test_invalid_create_value(self):
        """Invalid create value should produce error."""
        constraints = [
            {"edge_type": "MITIGATES", "create": "invalid"}
        ]

        errors = validate_edge_constraints(constraints)

        assert len(errors) == 1
        assert "create must be 'auto' or 'never'" in errors[0]

    def test_invalid_direction_value(self):
        """Invalid direction value should produce error."""
        constraints = [
            {"edge_type": "MITIGATES", "direction": "invalid"}
        ]

        errors = validate_edge_constraints(constraints)

        assert len(errors) == 1
        assert "direction must be" in errors[0]

    def test_valid_direction_values(self):
        """Valid direction values should pass."""
        constraints = [
            {"edge_type": "MITIGATES", "direction": "outgoing"},
            {"edge_type": "IS_INSTANCE", "direction": "incoming"},
            {"edge_type": "LEADS_TO", "direction": "both"}
        ]

        errors = validate_edge_constraints(constraints)

        assert errors == []

    def test_invalid_search_config(self):
        """Invalid search config should produce error."""
        constraints = [
            {"edge_type": "MITIGATES", "search": "invalid"}
        ]

        errors = validate_edge_constraints(constraints)

        assert len(errors) == 1
        assert "search: must be a dictionary" in errors[0]

    def test_invalid_when_clause(self):
        """Invalid when clause should produce error."""
        constraints = [
            {"edge_type": "MITIGATES", "when": "invalid"}
        ]

        errors = validate_edge_constraints(constraints)

        assert len(errors) == 1
        assert "when: must be a dictionary" in errors[0]


class TestDeepTrustEdgeScenarios:
    """Test DeepTrust security domain edge scenarios."""

    @pytest.mark.asyncio
    async def test_mitigates_controlled_vocabulary(
        self,
        mock_memory_graph,
        controlled_vocabulary_edge_constraint
    ):
        """MITIGATES edge should only link to existing TacticDef."""
        source_node = {
            "type": "SecurityBehavior",
            "properties": {"id": "SB080", "name": "Verify Identity"}
        }
        target_node = {
            "type": "TacticDef",
            "properties": {"id": "TA0005", "name": "Defense Evasion"}
        }
        constraints = [controlled_vocabulary_edge_constraint]

        # Mock finds existing TacticDef
        mock_memory_graph.find_node_by_property = AsyncMock(return_value={
            "id": "existing_tactic",
            "type": "TacticDef",
            "properties": {"id": "TA0005", "name": "Defense Evasion"}
        })

        should_create, final_target, props = await apply_edge_constraints(
            source_node=source_node,
            target_node=target_node,
            edge_type="MITIGATES",
            edge_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_edge_properties={}
        )

        assert should_create is True
        assert final_target["properties"]["id"] == "TA0005"

    @pytest.mark.asyncio
    async def test_is_instance_auto_create(
        self,
        mock_memory_graph,
        auto_create_edge_constraint
    ):
        """IS_INSTANCE edge with auto create should allow creation."""
        source_node = {
            "type": "CallerTactic",
            "properties": {"tactic_name": "Lost Phone Claim"}
        }
        target_node = {
            "type": "TacticDef",
            "properties": {"id": "TA0005"}
        }
        constraints = [auto_create_edge_constraint]

        mock_memory_graph.find_node_by_property = AsyncMock(return_value=None)

        should_create, final_target, props = await apply_edge_constraints(
            source_node=source_node,
            target_node=target_node,
            edge_type="IS_INSTANCE",
            edge_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_edge_properties={"confidence": 0.95}
        )

        assert should_create is True

    @pytest.mark.asyncio
    async def test_mitigates_blocks_unknown_tactic(self, mock_memory_graph):
        """MITIGATES edge should be blocked for unknown TacticDef."""
        source_node = {"type": "SecurityBehavior", "properties": {"id": "SB080"}}
        target_node = {"type": "TacticDef", "properties": {"id": "TA9999"}}  # Unknown
        constraints = [
            {
                "edge_type": "MITIGATES",
                "source_type": "SecurityBehavior",
                "target_type": "TacticDef",
                "create": "never",
                "search": {"properties": [{"name": "id", "mode": "exact"}]}
            }
        ]

        mock_memory_graph.find_node_by_property = AsyncMock(return_value=None)

        should_create, final_target, props = await apply_edge_constraints(
            source_node=source_node,
            target_node=target_node,
            edge_type="MITIGATES",
            edge_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_edge_properties={}
        )

        assert should_create is False
        assert final_target is None


class TestDocumentedExamples:
    """Test examples from DX documentation."""

    @pytest.mark.asyncio
    async def test_edge_create_never(self, mock_memory_graph):
        """Test edge constraint with create='never'."""
        source_node = {"type": "SecurityBehavior", "properties": {}}
        target_node = {"type": "TacticDef", "properties": {"name": "Defense Evasion"}}
        constraints = [
            {
                "edge_type": "MITIGATES",
                "source_type": "SecurityBehavior",
                "target_type": "TacticDef",
                "create": "never",
                "search": {"properties": [{"name": "name", "mode": "semantic", "threshold": 0.90}]}
            }
        ]

        # Semantic search finds existing
        mock_memory_graph.find_node_by_semantic_match = AsyncMock(return_value={
            "id": "tactic_1",
            "type": "TacticDef",
            "properties": {"id": "TA0005", "name": "Defense Evasion"}
        })

        should_create, final_target, props = await apply_edge_constraints(
            source_node=source_node,
            target_node=target_node,
            edge_type="MITIGATES",
            edge_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_edge_properties={}
        )

        assert should_create is True
        assert final_target is not None

    @pytest.mark.asyncio
    async def test_edge_with_when_condition(self, mock_memory_graph):
        """Test edge constraint with when condition."""
        source_node = {"type": "TacticDef", "properties": {}}
        target_node = {"type": "Impact", "properties": {"name": "System Compromise"}}
        constraints = [
            {
                "edge_type": "LEADS_TO",
                "when": {"severity": "critical"}
            }
        ]

        # Test with critical severity
        extracted_critical = {"severity": "critical"}
        should_create, _, _ = await apply_edge_constraints(
            source_node=source_node,
            target_node=target_node,
            edge_type="LEADS_TO",
            edge_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_edge_properties=extracted_critical
        )

        # Constraint applied (default auto)
        assert should_create is True

        # Test with low severity
        extracted_low = {"severity": "low"}
        should_create, _, _ = await apply_edge_constraints(
            source_node=source_node,
            target_node=target_node,
            edge_type="LEADS_TO",
            edge_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_edge_properties=extracted_low
        )

        # Condition not met, constraint skipped
        assert should_create is True
