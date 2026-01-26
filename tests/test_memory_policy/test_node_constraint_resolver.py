"""
Tests for node constraint resolver - applying node constraints during graph generation.

Tests the apply_node_constraints function which handles:
- Finding applicable constraints by node_type
- Evaluating 'when' conditions
- Searching for existing nodes (exact, semantic, fuzzy, via_relationship)
- Applying create policy (auto vs never)
- Setting property values

Run with: pytest tests/test_memory_policy/test_node_constraint_resolver.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from services.node_constraint_resolver import (
    apply_node_constraints,
    _find_applicable_constraint,
    _apply_set_values,
    get_node_constraints_for_type,
    validate_node_constraints
)


class TestFindApplicableConstraint:
    """Test finding the most applicable constraint for a node."""

    def test_exact_type_match(self):
        """Should find constraint with exact node_type match."""
        constraints = [
            {"node_type": "TacticDef", "create": "never"},
            {"node_type": "SecurityBehavior", "create": "never"}
        ]

        result = _find_applicable_constraint(
            node_type="TacticDef",
            node_constraints=constraints
        )

        assert result is not None
        assert result["node_type"] == "TacticDef"
        assert result["create"] == "never"

    def test_no_matching_constraint(self):
        """Should return None when no constraint matches."""
        constraints = [
            {"node_type": "TacticDef", "create": "never"}
        ]

        result = _find_applicable_constraint(
            node_type="CallerTactic",
            node_constraints=constraints
        )

        assert result is None

    def test_first_match_wins(self):
        """Should return first matching constraint."""
        constraints = [
            {"node_type": "TacticDef", "create": "never"},
            {"node_type": "TacticDef", "create": "auto"}  # Second match
        ]

        result = _find_applicable_constraint(
            node_type="TacticDef",
            node_constraints=constraints
        )

        assert result["create"] == "never"  # First match

    def test_empty_constraints_returns_none(self):
        """Should return None for empty constraints list."""
        result = _find_applicable_constraint(
            node_type="TacticDef",
            node_constraints=[]
        )

        assert result is None

    def test_wildcard_constraint_no_node_type(self):
        """Constraint without node_type should match any node."""
        constraints = [
            {"create": "auto"}  # No node_type - wildcard
        ]

        result = _find_applicable_constraint(
            node_type="AnyType",
            node_constraints=constraints
        )

        assert result is not None
        assert result["create"] == "auto"

    def test_specific_before_wildcard(self):
        """Specific type constraint should be found before wildcard."""
        constraints = [
            {"node_type": "TacticDef", "create": "never"},
            {"create": "auto"}  # Wildcard
        ]

        result = _find_applicable_constraint(
            node_type="TacticDef",
            node_constraints=constraints
        )

        # First match wins - the specific TacticDef constraint
        assert result["node_type"] == "TacticDef"
        assert result["create"] == "never"


class TestApplyNodeConstraints:
    """Test the full node constraint application flow."""

    @pytest.mark.asyncio
    async def test_no_constraints_allows_creation(self, mock_memory_graph):
        """No constraints should allow node creation with extracted properties."""
        node = {"type": "CallerTactic", "properties": {"tactic_name": "Defense Evasion"}}
        extracted = {"tactic_name": "Defense Evasion", "severity": "high"}

        should_create, existing, props = await apply_node_constraints(
            node=node,
            node_type="CallerTactic",
            node_constraints=[],
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        assert should_create is True
        assert existing is None
        assert props == extracted

    @pytest.mark.asyncio
    async def test_no_matching_constraint_allows_creation(self, mock_memory_graph):
        """No matching constraint should allow node creation."""
        node = {"type": "CallerTactic", "properties": {}}
        constraints = [
            {"node_type": "TacticDef", "create": "never"}  # Different type
        ]
        extracted = {"tactic_name": "Defense Evasion"}

        should_create, existing, props = await apply_node_constraints(
            node=node,
            node_type="CallerTactic",
            node_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        assert should_create is True
        assert existing is None
        assert props == extracted

    @pytest.mark.asyncio
    async def test_create_never_no_existing_blocks_creation(self, mock_memory_graph):
        """create='never' with no existing node should block creation."""
        node = {"type": "TacticDef", "properties": {"id": "TA9999"}}  # Unknown ID
        constraints = [
            {
                "node_type": "TacticDef",
                "create": "never",
                "search": {"properties": [{"name": "id", "mode": "exact"}]}
            }
        ]
        extracted = {"id": "TA9999", "name": "Unknown Tactic"}

        # Mock returns None (no existing node)
        mock_memory_graph.find_node_by_property = AsyncMock(return_value=None)

        should_create, existing, props = await apply_node_constraints(
            node=node,
            node_type="TacticDef",
            node_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        assert should_create is False
        assert existing is None
        assert props is None

    @pytest.mark.asyncio
    async def test_create_never_with_existing_returns_existing(self, mock_memory_graph):
        """create='never' with existing node should return existing."""
        node = {"type": "TacticDef", "properties": {"id": "TA0005"}}
        constraints = [
            {
                "node_type": "TacticDef",
                "create": "never",
                "search": {"properties": [{"name": "id", "mode": "exact"}]}
            }
        ]
        extracted = {"id": "TA0005", "name": "Defense Evasion"}

        # mock_memory_graph fixture returns existing node for TA0005

        should_create, existing, props = await apply_node_constraints(
            node=node,
            node_type="TacticDef",
            node_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        assert should_create is False
        assert existing is not None
        assert existing["properties"]["id"] == "TA0005"
        assert props is None  # No properties to set

    @pytest.mark.asyncio
    async def test_create_auto_with_existing_returns_existing(self, mock_memory_graph):
        """create='auto' with existing node should return existing with merged props."""
        node = {"type": "TacticDef", "properties": {"id": "TA0005"}}
        constraints = [
            {
                "node_type": "TacticDef",
                "create": "auto",
                "search": {"properties": [{"name": "id", "mode": "exact"}]}
            }
        ]
        extracted = {"id": "TA0005", "name": "Defense Evasion", "severity": "high"}

        should_create, existing, props = await apply_node_constraints(
            node=node,
            node_type="TacticDef",
            node_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        assert should_create is False
        assert existing is not None
        assert props == extracted  # Returns merged properties

    @pytest.mark.asyncio
    async def test_create_auto_no_existing_allows_creation(self, mock_memory_graph):
        """create='auto' with no existing node should allow creation."""
        node = {"type": "CallerTactic", "properties": {}}
        constraints = [
            {
                "node_type": "CallerTactic",
                "create": "auto",
                "search": {"properties": [{"name": "tactic_name", "mode": "semantic"}]}
            }
        ]
        extracted = {"tactic_name": "New Tactic", "severity": "low"}

        mock_memory_graph.find_node_by_semantic_match = AsyncMock(return_value=None)

        should_create, existing, props = await apply_node_constraints(
            node=node,
            node_type="CallerTactic",
            node_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        assert should_create is True
        assert existing is None
        assert props == extracted

    @pytest.mark.asyncio
    async def test_when_condition_blocks_constraint(self, mock_memory_graph):
        """When condition not met should skip constraint, allow creation."""
        node = {"type": "CallerTactic", "properties": {}}
        constraints = [
            {
                "node_type": "CallerTactic",
                "create": "never",  # Would block creation
                "when": {"severity": "critical"}  # Condition not met
            }
        ]
        extracted = {"tactic_name": "Test", "severity": "low"}  # Not critical

        should_create, existing, props = await apply_node_constraints(
            node=node,
            node_type="CallerTactic",
            node_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        # When condition not met, constraint not applied, creation allowed
        assert should_create is True
        assert props == extracted

    @pytest.mark.asyncio
    async def test_when_condition_allows_constraint(self, mock_memory_graph):
        """When condition met should apply constraint."""
        node = {"type": "CallerTactic", "properties": {}}
        constraints = [
            {
                "node_type": "CallerTactic",
                "create": "never",
                "when": {"severity": "critical"}
            }
        ]
        extracted = {"tactic_name": "Test", "severity": "critical"}

        mock_memory_graph.find_node_by_property = AsyncMock(return_value=None)

        should_create, existing, props = await apply_node_constraints(
            node=node,
            node_type="CallerTactic",
            node_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        # When condition met, create='never' blocks creation
        assert should_create is False

    @pytest.mark.asyncio
    async def test_semantic_search_finds_existing(self, mock_memory_graph):
        """Semantic search should find existing node."""
        node = {"type": "TacticDef", "properties": {"name": "Defense Evasion"}}
        constraints = [
            {
                "node_type": "TacticDef",
                "create": "never",
                "search": {"properties": [{"name": "name", "mode": "semantic", "threshold": 0.90}]}
            }
        ]
        extracted = {"name": "Defense Evasion"}

        should_create, existing, props = await apply_node_constraints(
            node=node,
            node_type="TacticDef",
            node_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        assert should_create is False
        assert existing is not None  # Found via semantic search


class TestApplySetValues:
    """Test property value application from 'set' clause."""

    def test_exact_value_override(self):
        """Exact values should override extracted values."""
        constraint = {
            "set": {"status": "completed", "priority": "high"}
        }
        extracted = {"title": "Task 1", "status": "pending"}

        result = _apply_set_values(constraint, extracted)

        assert result["title"] == "Task 1"  # Preserved
        assert result["status"] == "completed"  # Overridden
        assert result["priority"] == "high"  # Added

    def test_auto_mode_preserves_extracted(self):
        """mode='auto' should preserve extracted value."""
        constraint = {
            "set": {"status": {"mode": "auto"}}
        }
        extracted = {"title": "Task 1", "status": "in_progress"}

        result = _apply_set_values(constraint, extracted)

        assert result["title"] == "Task 1"
        assert result["status"] == "in_progress"  # Preserved

    def test_auto_mode_no_existing_value(self):
        """mode='auto' with no existing value should not add property."""
        constraint = {
            "set": {"priority": {"mode": "auto"}}
        }
        extracted = {"title": "Task 1"}  # No priority

        result = _apply_set_values(constraint, extracted)

        assert result["title"] == "Task 1"
        # priority not added (auto mode, no extracted value)

    def test_mixed_exact_and_auto(self):
        """Mixed exact values and auto mode should work."""
        constraint = {
            "set": {
                "workspace_id": "ws_123",  # Exact
                "status": {"mode": "auto"}  # Auto
            }
        }
        extracted = {"title": "Task 1", "status": "active"}

        result = _apply_set_values(constraint, extracted)

        assert result["workspace_id"] == "ws_123"  # Exact override
        assert result["status"] == "active"  # Auto preserved

    def test_no_set_values(self):
        """No set values should return extracted unchanged."""
        constraint = {}
        extracted = {"title": "Task 1", "status": "pending"}

        result = _apply_set_values(constraint, extracted)

        assert result == extracted

    def test_none_extracted_properties(self):
        """None extracted properties should be handled."""
        constraint = {
            "set": {"status": "completed"}
        }

        result = _apply_set_values(constraint, None)

        assert result["status"] == "completed"

    def test_boolean_exact_value(self):
        """Boolean exact values should work."""
        constraint = {
            "set": {"verified": True, "archived": False}
        }
        extracted = {"title": "Task 1"}

        result = _apply_set_values(constraint, extracted)

        assert result["verified"] is True
        assert result["archived"] is False

    def test_integer_exact_value(self):
        """Integer exact values should work."""
        constraint = {
            "set": {"priority_score": 5}
        }
        extracted = {"title": "Task 1"}

        result = _apply_set_values(constraint, extracted)

        assert result["priority_score"] == 5

    def test_list_exact_value(self):
        """List exact values should work."""
        constraint = {
            "set": {"tags": ["urgent", "bug"]}
        }
        extracted = {"title": "Task 1"}

        result = _apply_set_values(constraint, extracted)

        assert result["tags"] == ["urgent", "bug"]


class TestTextModeSetValues:
    """Test text_mode options: replace, append, merge."""

    def test_text_mode_replace_default(self):
        """text_mode='replace' (default) should overwrite existing value."""
        constraint = {
            "set": {"notes": {"mode": "auto"}}  # Default text_mode is 'replace'
        }
        extracted = {"notes": "New notes content"}
        existing = {"notes": "Old notes content"}

        result = _apply_set_values(constraint, extracted, existing)

        # Replace: extracted value replaces existing
        assert result["notes"] == "New notes content"

    def test_text_mode_replace_explicit(self):
        """Explicit text_mode='replace' should overwrite existing value."""
        constraint = {
            "set": {"status": {"mode": "auto", "text_mode": "replace"}}
        }
        extracted = {"status": "completed"}
        existing = {"status": "in_progress"}

        result = _apply_set_values(constraint, extracted, existing)

        assert result["status"] == "completed"

    def test_text_mode_append_string(self):
        """text_mode='append' should add to end of existing string."""
        constraint = {
            "set": {"notes": {"mode": "auto", "text_mode": "append"}}
        }
        extracted = {"notes": "Update: needs security review before deploy"}
        existing = {"notes": "Original task notes"}

        result = _apply_set_values(constraint, extracted, existing)

        assert result["notes"] == "Original task notes\nUpdate: needs security review before deploy"

    def test_text_mode_append_list(self):
        """text_mode='append' should add to existing list."""
        constraint = {
            "set": {"comments": {"mode": "auto", "text_mode": "append"}}
        }
        extracted = {"comments": "New comment"}
        existing = {"comments": ["First comment", "Second comment"]}

        result = _apply_set_values(constraint, extracted, existing)

        assert result["comments"] == ["First comment", "Second comment", "New comment"]

    def test_text_mode_append_no_existing(self):
        """text_mode='append' with no existing value should use extracted as-is."""
        constraint = {
            "set": {"notes": {"mode": "auto", "text_mode": "append"}}
        }
        extracted = {"notes": "First notes"}
        existing = {}  # No existing notes

        result = _apply_set_values(constraint, extracted, existing)

        assert result["notes"] == "First notes"

    def test_text_mode_append_no_extracted(self):
        """text_mode='append' with no extracted value should keep existing."""
        constraint = {
            "set": {"notes": {"mode": "auto", "text_mode": "append"}}
        }
        extracted = {}  # No new notes
        existing = {"notes": "Original notes"}

        result = _apply_set_values(constraint, extracted, existing)

        assert result["notes"] == "Original notes"

    def test_text_mode_merge(self):
        """text_mode='merge' should mark for intelligent LLM merge."""
        constraint = {
            "set": {"summary": {"mode": "auto", "text_mode": "merge"}}
        }
        extracted = {"summary": "Task is now complete and deployed"}
        existing = {"summary": "Task fixes authentication bug"}

        result = _apply_set_values(constraint, extracted, existing)

        # Merge creates a merge marker for LLM processing
        assert result["summary"]["_merge"] is True
        assert result["summary"]["existing"] == "Task fixes authentication bug"
        assert result["summary"]["new"] == "Task is now complete and deployed"

    def test_text_mode_merge_no_existing(self):
        """text_mode='merge' with no existing value should use extracted as-is."""
        constraint = {
            "set": {"summary": {"mode": "auto", "text_mode": "merge"}}
        }
        extracted = {"summary": "New summary content"}
        existing = {}  # No existing summary

        result = _apply_set_values(constraint, extracted, existing)

        # No existing to merge with, just use extracted
        assert result["summary"] == "New summary content"

    def test_text_mode_merge_no_extracted(self):
        """text_mode='merge' with no extracted value should keep existing."""
        constraint = {
            "set": {"summary": {"mode": "auto", "text_mode": "merge"}}
        }
        extracted = {}  # No new summary
        existing = {"summary": "Existing summary"}

        result = _apply_set_values(constraint, extracted, existing)

        assert result["summary"] == "Existing summary"

    def test_mixed_text_modes(self):
        """Multiple properties with different text_modes should work."""
        constraint = {
            "set": {
                "status": {"mode": "auto"},  # replace (default)
                "notes": {"mode": "auto", "text_mode": "append"},
                "summary": {"mode": "auto", "text_mode": "merge"},
                "project_id": "PROJ-456"  # exact value
            }
        }
        extracted = {
            "status": "in_progress",
            "notes": "Update: needs security review",
            "summary": "Task is now in progress"
        }
        existing = {
            "status": "pending",
            "notes": "Original notes",
            "summary": "Task fixes auth bug"
        }

        result = _apply_set_values(constraint, extracted, existing)

        # Replace (default)
        assert result["status"] == "in_progress"
        # Append
        assert result["notes"] == "Original notes\nUpdate: needs security review"
        # Merge
        assert result["summary"]["_merge"] is True
        assert result["summary"]["existing"] == "Task fixes auth bug"
        assert result["summary"]["new"] == "Task is now in progress"
        # Exact
        assert result["project_id"] == "PROJ-456"

    def test_real_world_task_follow_up(self):
        """Real-world scenario: Follow-up on task with notes append."""
        # From DX documentation: Follow-up on a task - add new notes without overwriting
        constraint = {
            "set": {
                "status": {"mode": "auto"},  # LLM extracts "in progress"
                "notes": {"mode": "auto", "text_mode": "append"},
                "project_id": "PROJ-456"  # Exact value
            }
        }
        extracted = {
            "title": "Authentication Bug",
            "status": "in_progress",
            "notes": "Update: needs security review before deploy"
        }
        existing = {
            "title": "Authentication Bug",
            "status": "pending",
            "notes": "Initial bug report: login fails for certain users"
        }

        result = _apply_set_values(constraint, extracted, existing)

        assert result["status"] == "in_progress"
        assert "Initial bug report" in result["notes"]
        assert "Update: needs security review" in result["notes"]
        assert result["project_id"] == "PROJ-456"


class TestGetNodeConstraintsForType:
    """Test getting all constraints for a node type."""

    def test_exact_match(self):
        """Should return constraints matching exact type."""
        constraints = [
            {"node_type": "TacticDef", "create": "never"},
            {"node_type": "SecurityBehavior", "create": "never"},
            {"node_type": "TacticDef", "create": "auto"}
        ]

        result = get_node_constraints_for_type("TacticDef", constraints)

        assert len(result) == 2
        for c in result:
            assert c["node_type"] == "TacticDef"

    def test_wildcard_constraints(self):
        """Constraints without node_type should match any type."""
        constraints = [
            {"node_type": "TacticDef", "create": "never"},
            {"create": "auto"}  # Wildcard
        ]

        result = get_node_constraints_for_type("CallerTactic", constraints)

        # Only wildcard matches CallerTactic
        assert len(result) == 1
        assert "node_type" not in result[0]

    def test_no_matches(self):
        """Should return empty list when no matches."""
        constraints = [
            {"node_type": "TacticDef", "create": "never"}
        ]

        result = get_node_constraints_for_type("CallerTactic", constraints)

        assert result == []


class TestValidateNodeConstraints:
    """Test node constraint validation."""

    def test_valid_constraint(self):
        """Valid constraint should pass validation."""
        constraints = [
            {
                "node_type": "TacticDef",
                "create": "never",
                "search": {"properties": [{"name": "id", "mode": "exact"}]}
            }
        ]

        errors = validate_node_constraints(constraints)

        assert errors == []

    def test_missing_node_type(self):
        """Missing node_type should produce error."""
        constraints = [
            {"create": "never"}  # No node_type
        ]

        errors = validate_node_constraints(constraints)

        assert len(errors) == 1
        assert "node_type is required" in errors[0]

    def test_invalid_create_value(self):
        """Invalid create value should produce error."""
        constraints = [
            {"node_type": "TacticDef", "create": "invalid"}
        ]

        errors = validate_node_constraints(constraints)

        assert len(errors) == 1
        assert "create must be 'auto' or 'never'" in errors[0]

    def test_invalid_search_config(self):
        """Invalid search config should produce error."""
        constraints = [
            {"node_type": "TacticDef", "search": "invalid"}  # Should be dict
        ]

        errors = validate_node_constraints(constraints)

        assert len(errors) == 1
        assert "search: must be a dictionary" in errors[0]

    def test_invalid_search_properties(self):
        """Invalid search properties should produce error."""
        constraints = [
            {"node_type": "TacticDef", "search": {"properties": "invalid"}}
        ]

        errors = validate_node_constraints(constraints)

        assert len(errors) == 1
        assert "search.properties: must be a list" in errors[0]

    def test_invalid_when_clause(self):
        """Invalid when clause should produce error."""
        constraints = [
            {"node_type": "TacticDef", "when": "invalid"}  # Should be dict
        ]

        errors = validate_node_constraints(constraints)

        assert len(errors) == 1
        assert "when: must be a dictionary" in errors[0]

    def test_valid_when_with_operators(self):
        """Valid when clause with operators should pass."""
        constraints = [
            {
                "node_type": "TacticDef",
                "create": "auto",
                "when": {
                    "_and": [
                        {"severity": "critical"},
                        {"_not": {"acknowledged": True}}
                    ]
                }
            }
        ]

        errors = validate_node_constraints(constraints)

        assert errors == []

    def test_unknown_when_operator(self):
        """Unknown when operator should produce error."""
        constraints = [
            {
                "node_type": "TacticDef",
                "when": {"_unknown": []}
            }
        ]

        errors = validate_node_constraints(constraints)

        assert len(errors) == 1
        assert "unknown operator '_unknown'" in errors[0]

    def test_invalid_nested_when(self):
        """Invalid nested when should produce error."""
        constraints = [
            {
                "node_type": "TacticDef",
                "when": {
                    "_and": [
                        "invalid"  # Should be dict
                    ]
                }
            }
        ]

        errors = validate_node_constraints(constraints)

        assert len(errors) == 1


class TestDeepTrustScenarios:
    """Test DeepTrust security domain scenarios."""

    @pytest.mark.asyncio
    async def test_controlled_vocabulary_tactic_def(
        self,
        mock_memory_graph,
        controlled_vocabulary_node_constraint
    ):
        """TacticDef controlled vocabulary should require existing node."""
        node = {"type": "TacticDef", "properties": {"id": "TA0005"}}
        constraints = [controlled_vocabulary_node_constraint]
        extracted = {"id": "TA0005", "name": "Defense Evasion"}

        should_create, existing, props = await apply_node_constraints(
            node=node,
            node_type="TacticDef",
            node_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        assert should_create is False
        assert existing is not None
        assert existing["properties"]["id"] == "TA0005"

    @pytest.mark.asyncio
    async def test_dynamic_entity_caller_tactic(
        self,
        mock_memory_graph,
        auto_create_node_constraint
    ):
        """CallerTactic dynamic entity should allow creation."""
        node = {"type": "CallerTactic", "properties": {}}
        constraints = [auto_create_node_constraint]
        extracted = {"tactic_name": "New Attack Vector", "severity": "high"}

        mock_memory_graph.find_node_by_semantic_match = AsyncMock(return_value=None)

        should_create, existing, props = await apply_node_constraints(
            node=node,
            node_type="CallerTactic",
            node_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        assert should_create is True
        assert existing is None

    @pytest.mark.asyncio
    async def test_conditional_alert_on_critical_tactic(
        self,
        mock_memory_graph,
        conditional_node_constraint
    ):
        """Critical CallerTactic should trigger alert_security_team set."""
        node = {"type": "CallerTactic", "properties": {}}
        constraints = [conditional_node_constraint]
        extracted = {"tactic_name": "Critical Attack", "severity": "critical"}

        should_create, existing, props = await apply_node_constraints(
            node=node,
            node_type="CallerTactic",
            node_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        # Condition met, set values applied
        assert props["alert_security_team"] is True

    @pytest.mark.asyncio
    async def test_conditional_no_alert_on_non_critical(
        self,
        mock_memory_graph,
        conditional_node_constraint
    ):
        """Non-critical CallerTactic should not trigger alert."""
        node = {"type": "CallerTactic", "properties": {}}
        constraints = [conditional_node_constraint]
        extracted = {"tactic_name": "Minor Issue", "severity": "low"}

        should_create, existing, props = await apply_node_constraints(
            node=node,
            node_type="CallerTactic",
            node_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        # Condition not met, original props returned
        assert "alert_security_team" not in props

    @pytest.mark.asyncio
    async def test_via_relationship_finds_security_behavior(
        self,
        mock_memory_graph,
        via_relationship_node_constraint
    ):
        """Via relationship should find SecurityBehavior via TacticDef."""
        node = {"type": "SecurityBehavior", "properties": {"name": "Verify Identity"}}
        constraints = [via_relationship_node_constraint]
        extracted = {"name": "Verify Identity", "trigger_context": "Defense Evasion"}

        # Setup: First find TacticDef via semantic search
        mock_memory_graph.find_node_by_semantic_match = AsyncMock(side_effect=[
            None,  # No direct semantic match on SecurityBehavior.name
            {  # TacticDef found via target_search
                "id": "existing_tactic_1",
                "type": "TacticDef",
                "properties": {"name": "Defense Evasion"}
            }
        ])

        # Then find SecurityBehavior via relationship
        mock_memory_graph.find_node_via_relationship = AsyncMock(return_value={
            "id": "existing_behavior_1",
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

        # create='never' with existing node found via relationship
        assert should_create is False
        assert existing is not None


class TestDocumentedExamples:
    """Test examples from DX documentation."""

    @pytest.mark.asyncio
    async def test_link_to_task_semantic(self, mock_memory_graph):
        """Test: link_to="Task:title" - semantic search on title."""
        # Simulates the expanded form of link_to="Task:title"
        # Note: node.properties must contain the search value for the search to work
        node = {"type": "Task", "properties": {"title": "Authentication Bug Fix"}}
        constraints = [
            {
                "node_type": "Task",
                "search": {"properties": [{"name": "title", "mode": "semantic"}]}
            }
        ]
        extracted = {"title": "Authentication Bug Fix"}

        mock_memory_graph.find_node_by_semantic_match = AsyncMock(return_value={
            "id": "task_1",
            "type": "Task",
            "properties": {"title": "Authentication Bug"}
        })

        should_create, existing, props = await apply_node_constraints(
            node=node,
            node_type="Task",
            node_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        # Found existing via semantic match
        assert should_create is False
        assert existing is not None

    @pytest.mark.asyncio
    async def test_link_to_task_with_set(self, mock_memory_graph):
        """Test: link_to={"Task:title": {"set": {"status": "completed"}}}"""
        # Note: node.properties must contain the search value for the search to work
        node = {"type": "Task", "properties": {"title": "Auth Bug"}}
        constraints = [
            {
                "node_type": "Task",
                "search": {"properties": [{"name": "title", "mode": "semantic"}]},
                "set": {"status": "completed"}
            }
        ]
        extracted = {"title": "Auth Bug"}

        mock_memory_graph.find_node_by_semantic_match = AsyncMock(return_value={
            "id": "task_1",
            "type": "Task",
            "properties": {"title": "Auth Bug", "status": "in_progress"}
        })

        should_create, existing, props = await apply_node_constraints(
            node=node,
            node_type="Task",
            node_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        assert existing is not None
        assert props["status"] == "completed"  # Set value applied

    @pytest.mark.asyncio
    async def test_link_to_person_create_never(self, mock_memory_graph):
        """Test: link_to={"Person:email": {"create": "never"}}"""
        node = {"type": "Person", "properties": {}}
        constraints = [
            {
                "node_type": "Person",
                "search": {"properties": [{"name": "email", "mode": "exact"}]},
                "create": "never"
            }
        ]
        extracted = {"email": "unknown@example.com"}

        mock_memory_graph.find_node_by_property = AsyncMock(return_value=None)

        should_create, existing, props = await apply_node_constraints(
            node=node,
            node_type="Person",
            node_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted
        )

        # create='never' blocks creation when not found
        assert should_create is False
        assert existing is None
        assert props is None

    @pytest.mark.asyncio
    async def test_conditional_task_update(self, mock_memory_graph):
        """Test: link_to={"Task:title": {"when": {"priority": "high"}, "set": {"status": "completed"}}}"""
        node = {"type": "Task", "properties": {}}
        constraints = [
            {
                "node_type": "Task",
                "search": {"properties": [{"name": "title", "mode": "semantic"}]},
                "when": {"priority": "high"},
                "set": {"status": "completed"}
            }
        ]

        mock_memory_graph.find_node_by_semantic_match = AsyncMock(return_value=None)

        # Test with high priority
        extracted_high = {"title": "Critical Bug", "priority": "high"}
        should_create, _, props = await apply_node_constraints(
            node=node,
            node_type="Task",
            node_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted_high
        )

        # Condition met, set applied
        assert props["status"] == "completed"

        # Test with low priority
        extracted_low = {"title": "Minor Bug", "priority": "low"}
        should_create, _, props = await apply_node_constraints(
            node=node,
            node_type="Task",
            node_constraints=constraints,
            memory_graph=mock_memory_graph,
            extracted_node_properties=extracted_low
        )

        # Condition not met, original props
        assert "status" not in props or props.get("status") != "completed"
