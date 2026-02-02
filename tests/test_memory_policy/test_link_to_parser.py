"""
Unit tests for Link-To DSL Parser.

Tests the parsing of link_to shorthand DSL into NodeConstraint and EdgeConstraint objects.

The link_to DSL provides a concise way to specify:
- Node constraints: "Task:title", "Task:id=TASK-123", "Task:title~auth bug"
- Edge constraints: "Source->EDGE->Target:property"
- Via relationships: "Task.via(ASSIGNED_TO->Person:email)"
- Special references: "$this", "$previous", "$context:N"
"""

import pytest
import sys
sys.path.insert(0, '/Users/shawkatkabbara/Documents/GitHub/memory')

from services.link_to_parser import (
    parse_link_to,
    expand_link_to_to_policy,
    validate_link_to,
    LinkToParseResult,
    NODE_PATTERN,
    EDGE_ARROW_PATTERN,
    VIA_PATTERN,
    PROPERTY_PATTERN,
    CONTEXT_REF_PATTERN,
)


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_schema():
    """Sample schema for type inference tests."""
    return {
        "relationship_types": {
            "MITIGATES": {
                "allowed_source_types": ["SecurityBehavior"],
                "allowed_target_types": ["TacticDef"]
            },
            "ASSIGNED_TO": {
                "allowed_source_types": ["Task"],
                "allowed_target_types": ["Person", "Team"]
            }
        }
    }


# ============================================================================
# Node Constraint Parsing Tests
# ============================================================================

class TestNodeConstraintParsing:
    """Tests for parsing node constraint DSL syntax."""

    def test_simple_node_semantic_match(self):
        """Test simple node:property syntax (semantic match)."""
        result = parse_link_to("Task:title")

        assert not result.has_errors()
        assert len(result.node_constraints) == 1
        assert result.node_constraints[0]["node_type"] == "Task"
        assert result.node_constraints[0]["search"]["properties"][0]["name"] == "title"
        assert result.node_constraints[0]["search"]["properties"][0]["mode"] == "semantic"

    def test_node_exact_match(self):
        """Test node:property=value syntax (exact match)."""
        result = parse_link_to("Task:id=TASK-123")

        assert not result.has_errors()
        assert len(result.node_constraints) == 1
        constraint = result.node_constraints[0]
        assert constraint["node_type"] == "Task"
        assert constraint["search"]["properties"][0]["name"] == "id"
        assert constraint["search"]["properties"][0]["mode"] == "exact"
        assert constraint["search"]["properties"][0]["value"] == "TASK-123"

    def test_node_semantic_match_with_value(self):
        """Test node:property~value syntax (semantic match with value)."""
        result = parse_link_to("Task:title~auth bug")

        assert not result.has_errors()
        constraint = result.node_constraints[0]
        prop = constraint["search"]["properties"][0]
        assert prop["name"] == "title"
        assert prop["mode"] == "semantic"
        assert prop["value"] == "auth bug"

    def test_node_semantic_match_with_threshold(self):
        """Test node:property~@0.9 syntax (semantic with custom threshold)."""
        result = parse_link_to("Task:title~@0.9")

        assert not result.has_errors()
        constraint = result.node_constraints[0]
        prop = constraint["search"]["properties"][0]
        assert prop["name"] == "title"
        assert prop["mode"] == "semantic"
        assert prop["threshold"] == 0.9

    def test_node_semantic_match_with_threshold_and_value(self):
        """Test node:property~@0.9~value syntax."""
        result = parse_link_to("Task:title~@0.85~authentication")

        assert not result.has_errors()
        constraint = result.node_constraints[0]
        prop = constraint["search"]["properties"][0]
        assert prop["name"] == "title"
        assert prop["mode"] == "semantic"
        assert prop["threshold"] == 0.85
        assert prop["value"] == "authentication"

    def test_node_multiple_properties(self):
        """Test node:prop1,prop2 syntax (multiple properties)."""
        result = parse_link_to("Person:email,name")

        assert not result.has_errors()
        constraint = result.node_constraints[0]
        assert constraint["node_type"] == "Person"
        props = constraint["search"]["properties"]
        assert len(props) == 2
        assert props[0]["name"] == "email"
        assert props[1]["name"] == "name"

    def test_invalid_node_syntax_no_colon(self):
        """Test error for missing colon in node syntax."""
        result = parse_link_to("Task")

        assert result.has_errors()
        assert "Invalid node syntax" in result.errors[0]


# ============================================================================
# Edge Constraint Parsing Tests (Arrow Syntax)
# ============================================================================

class TestEdgeConstraintParsing:
    """Tests for parsing edge constraint DSL syntax (arrow syntax)."""

    def test_full_edge_path(self):
        """Test Source->EDGE->Target:property syntax."""
        result = parse_link_to("SecurityBehavior->MITIGATES->TacticDef:name")

        assert not result.has_errors()
        assert len(result.edge_constraints) == 1
        edge = result.edge_constraints[0]
        assert edge["source_type"] == "SecurityBehavior"
        assert edge["edge_type"] == "MITIGATES"
        assert edge["target_type"] == "TacticDef"
        assert edge["search"]["properties"][0]["name"] == "name"

    def test_edge_implicit_target(self, sample_schema):
        """Test Source->EDGE:property syntax with target inferred from schema."""
        result = parse_link_to("SecurityBehavior->MITIGATES:name", schema=sample_schema)

        assert not result.has_errors()
        edge = result.edge_constraints[0]
        assert edge["edge_type"] == "MITIGATES"
        # Target should be inferred from schema
        assert edge["target_type"] == "TacticDef"

    def test_edge_any_source(self):
        """Test ->EDGE->Target:property syntax (any source)."""
        result = parse_link_to("->MITIGATES->TacticDef:name")

        assert not result.has_errors()
        edge = result.edge_constraints[0]
        assert edge.get("source_type") is None
        assert edge["edge_type"] == "MITIGATES"
        assert edge["target_type"] == "TacticDef"

    def test_edge_with_exact_value(self):
        """Test edge with exact value match on target."""
        result = parse_link_to("Task->ASSIGNED_TO->Person:email=john@example.com")

        assert not result.has_errors()
        edge = result.edge_constraints[0]
        prop = edge["search"]["properties"][0]
        assert prop["name"] == "email"
        assert prop["mode"] == "exact"
        assert prop["value"] == "john@example.com"

    def test_edge_no_target_no_schema(self):
        """Test edge without target type and no schema (no inference)."""
        result = parse_link_to("Task->ASSIGNED_TO:email")

        assert not result.has_errors()
        edge = result.edge_constraints[0]
        assert edge["source_type"] == "Task"
        assert edge["edge_type"] == "ASSIGNED_TO"
        assert edge.get("target_type") is None  # Can't infer without schema


# ============================================================================
# Via Relationship Tests
# ============================================================================

class TestViaRelationship:
    """Tests for parsing via relationship DSL syntax."""

    def test_via_basic(self):
        """Test Type.via(EDGE->Target) syntax."""
        result = parse_link_to("Task.via(ASSIGNED_TO->Person:email)")

        assert not result.has_errors()
        assert len(result.node_constraints) == 1
        constraint = result.node_constraints[0]
        assert constraint["node_type"] == "Task"
        assert "via_relationship" in constraint["search"]
        via = constraint["search"]["via_relationship"][0]
        assert via["edge_type"] == "ASSIGNED_TO"
        assert via["target_type"] == "Person"
        assert via["direction"] == "outgoing"

    def test_via_with_exact_value(self):
        """Test Type.via(EDGE->Target:prop=value) syntax."""
        result = parse_link_to("Task.via(ASSIGNED_TO->Person:email=john@example.com)")

        assert not result.has_errors()
        constraint = result.node_constraints[0]
        via = constraint["search"]["via_relationship"][0]
        prop = via["target_search"]["properties"][0]
        assert prop["name"] == "email"
        assert prop["mode"] == "exact"
        assert prop["value"] == "john@example.com"


# ============================================================================
# Special References Tests
# ============================================================================

class TestSpecialRefs:
    """Tests for parsing special reference syntax."""

    def test_this_ref(self):
        """Test $this reference."""
        result = parse_link_to("$this")

        assert not result.has_errors()
        assert len(result.special_refs) == 1
        assert result.special_refs[0]["ref"] == "$this"

    def test_previous_ref(self):
        """Test $previous reference."""
        result = parse_link_to("$previous")

        assert not result.has_errors()
        assert len(result.special_refs) == 1
        assert result.special_refs[0]["ref"] == "$previous"

    def test_context_ref(self):
        """Test $context:N reference."""
        result = parse_link_to("$context:5")

        assert not result.has_errors()
        assert len(result.special_refs) == 1
        ref = result.special_refs[0]
        assert ref["ref"] == "$context"
        assert ref["count"] == 5

    def test_context_ref_large_number(self):
        """Test $context:N with larger number."""
        result = parse_link_to("$context:100")

        assert not result.has_errors()
        assert result.special_refs[0]["count"] == 100


# ============================================================================
# Dict Form Tests (with options)
# ============================================================================

class TestDictForm:
    """Tests for parsing dict form with options."""

    def test_dict_with_set_option(self):
        """Test dict form with 'set' option."""
        result = parse_link_to({
            "Task:title": {"set": {"status": "completed"}}
        })

        assert not result.has_errors()
        constraint = result.node_constraints[0]
        assert constraint["set"]["status"] == "completed"

    def test_dict_with_create_never(self):
        """Test dict form with 'create: never' option."""
        result = parse_link_to({
            "Person:email": {"create": "never"}
        })

        assert not result.has_errors()
        constraint = result.node_constraints[0]
        assert constraint["create"] == "never"

    def test_dict_with_when_clause(self):
        """Test dict form with 'when' conditional."""
        result = parse_link_to({
            "Task:title": {"when": {"priority": "high"}}
        })

        assert not result.has_errors()
        constraint = result.node_constraints[0]
        assert constraint["when"]["priority"] == "high"

    def test_dict_edge_with_options(self):
        """Test dict form edge constraint with options."""
        result = parse_link_to({
            "SecurityBehavior->MITIGATES->TacticDef:name": {"create": "never"}
        })

        assert not result.has_errors()
        edge = result.edge_constraints[0]
        assert edge["create"] == "never"

    def test_dict_multiple_keys(self):
        """Test dict form with multiple keys."""
        result = parse_link_to({
            "Task:title": {"set": {"status": "active"}},
            "Person:email": {"create": "never"}
        })

        assert not result.has_errors()
        assert len(result.node_constraints) == 2


# ============================================================================
# List Form Tests
# ============================================================================

class TestListForm:
    """Tests for parsing list form (multiple constraints)."""

    def test_list_of_strings(self):
        """Test list of string constraints."""
        result = parse_link_to(["Task:title", "Person:email"])

        assert not result.has_errors()
        assert len(result.node_constraints) == 2
        assert result.node_constraints[0]["node_type"] == "Task"
        assert result.node_constraints[1]["node_type"] == "Person"

    def test_list_mixed_nodes_and_edges(self):
        """Test list with both node and edge constraints."""
        result = parse_link_to([
            "Task:title",
            "SecurityBehavior->MITIGATES->TacticDef:name"
        ])

        assert not result.has_errors()
        assert len(result.node_constraints) == 1
        assert len(result.edge_constraints) == 1

    def test_list_with_special_refs(self):
        """Test list including special references."""
        result = parse_link_to(["Task:title", "$previous"])

        assert not result.has_errors()
        assert len(result.node_constraints) == 1
        assert len(result.special_refs) == 1

    def test_list_invalid_item_type(self):
        """Test error for non-string items in list."""
        result = parse_link_to(["Task:title", 123])

        assert result.has_errors()
        assert "must be strings" in result.errors[0]


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling and invalid syntax detection."""

    def test_invalid_type(self):
        """Test error for invalid link_to type."""
        result = parse_link_to(12345)

        assert result.has_errors()
        assert "must be str, list, or dict" in result.errors[0]

    def test_invalid_dict_key_type(self):
        """Test error for non-string dict keys."""
        result = parse_link_to({123: {"set": {}}})

        assert result.has_errors()
        assert "must be strings" in result.errors[0]

    def test_invalid_dict_value_type(self):
        """Test error for non-dict options value."""
        result = parse_link_to({"Task:title": "invalid"})

        assert result.has_errors()
        assert "must be a dict" in result.errors[0]

    def test_invalid_property_syntax(self):
        """Test error for invalid property syntax."""
        result = parse_link_to("Task:123invalid")

        assert result.has_errors()
        assert "Invalid property syntax" in result.errors[0]


# ============================================================================
# Expand to Policy Tests
# ============================================================================

class TestExpandToPolicy:
    """Tests for expand_link_to_to_policy function."""

    def test_expand_simple_node(self):
        """Test expanding simple node constraint to policy."""
        policy = expand_link_to_to_policy("Task:title")

        assert "node_constraints" in policy
        assert len(policy["node_constraints"]) == 1
        assert policy["node_constraints"][0]["node_type"] == "Task"

    def test_expand_simple_edge(self):
        """Test expanding simple edge constraint to policy."""
        policy = expand_link_to_to_policy("SecurityBehavior->MITIGATES->TacticDef:name")

        assert "edge_constraints" in policy
        assert len(policy["edge_constraints"]) == 1

    def test_merge_with_existing_policy(self):
        """Test merging link_to with existing memory_policy."""
        existing = {
            "mode": "auto",
            "node_constraints": [
                {"node_type": "Project", "create": "never"}
            ]
        }
        policy = expand_link_to_to_policy("Task:title", existing_policy=existing)

        # Should merge, not replace
        assert len(policy["node_constraints"]) == 2
        node_types = {c["node_type"] for c in policy["node_constraints"]}
        assert node_types == {"Project", "Task"}

    def test_expand_previous_ref(self):
        """Test expanding $previous creates relationship."""
        policy = expand_link_to_to_policy("$previous")

        assert "relationships" in policy
        assert len(policy["relationships"]) == 1
        rel = policy["relationships"][0]
        assert rel["source"] == "$this"
        assert rel["target"] == "$previous"
        assert rel["type"] == "FOLLOWS"

    def test_expand_context_ref(self):
        """Test expanding $context:N sets context_depth."""
        policy = expand_link_to_to_policy("$context:5")

        assert policy.get("context_depth") == 5

    def test_expand_raises_on_error(self):
        """Test that expand raises ValueError on parsing errors."""
        with pytest.raises(ValueError) as exc_info:
            expand_link_to_to_policy("InvalidSyntax")

        assert "parsing errors" in str(exc_info.value)


# ============================================================================
# Validation Tests
# ============================================================================

class TestValidation:
    """Tests for validate_link_to function."""

    def test_validate_valid_syntax(self):
        """Test validation returns empty list for valid syntax."""
        errors = validate_link_to("Task:title")
        assert errors == []

    def test_validate_invalid_syntax(self):
        """Test validation returns errors for invalid syntax."""
        errors = validate_link_to("InvalidSyntax")
        assert len(errors) > 0

    def test_validate_multiple_errors(self):
        """Test validation collects multiple errors."""
        errors = validate_link_to(["Valid:prop", 123, "Also:valid"])
        # Should have one error for the non-string item
        assert len(errors) == 1


# ============================================================================
# Regex Pattern Tests
# ============================================================================

class TestRegexPatterns:
    """Tests for regex patterns used in parsing."""

    def test_node_pattern_matches(self):
        """Test NODE_PATTERN matches valid node syntax."""
        assert NODE_PATTERN.match("Task:title")
        assert NODE_PATTERN.match("Person:email,name")
        assert NODE_PATTERN.match("MyCustomType123:prop")
        assert not NODE_PATTERN.match("Task")
        assert not NODE_PATTERN.match(":property")

    def test_edge_pattern_matches(self):
        """Test EDGE_ARROW_PATTERN matches valid edge syntax."""
        assert EDGE_ARROW_PATTERN.match("Source->EDGE->Target:prop")
        assert EDGE_ARROW_PATTERN.match("->EDGE->Target:prop")
        assert EDGE_ARROW_PATTERN.match("Source->EDGE:prop")
        assert EDGE_ARROW_PATTERN.match("Source->EDGE_TYPE->Target")

    def test_via_pattern_matches(self):
        """Test VIA_PATTERN matches valid via syntax."""
        assert VIA_PATTERN.match("Task.via(ASSIGNED_TO->Person:email)")
        assert VIA_PATTERN.match("Type.via(EDGE->Target)")
        assert not VIA_PATTERN.match("Task.via(invalid)")

    def test_property_pattern_matches(self):
        """Test PROPERTY_PATTERN matches valid property syntax."""
        # Simple name
        m = PROPERTY_PATTERN.match("title")
        assert m and m.group("name") == "title"

        # Exact match
        m = PROPERTY_PATTERN.match("id=TASK-123")
        assert m and m.group("operator") == "="

        # Semantic with value
        m = PROPERTY_PATTERN.match("title~auth bug")
        assert m and m.group("operator") == "~"

        # Semantic with threshold
        m = PROPERTY_PATTERN.match("title~@0.9")
        assert m and m.group("threshold") == "0.9"

    def test_context_ref_pattern_matches(self):
        """Test CONTEXT_REF_PATTERN matches valid context refs."""
        m = CONTEXT_REF_PATTERN.match("$context:5")
        assert m and m.group(1) == "5"

        m = CONTEXT_REF_PATTERN.match("$context:100")
        assert m and m.group(1) == "100"

        assert not CONTEXT_REF_PATTERN.match("$context")
        assert not CONTEXT_REF_PATTERN.match("$previous")


# ============================================================================
# LinkToParseResult Model Tests
# ============================================================================

class TestLinkToParseResultModel:
    """Tests for LinkToParseResult Pydantic model."""

    def test_result_is_pydantic_model(self):
        """Test that result is a proper Pydantic model."""
        result = LinkToParseResult()
        assert hasattr(result, 'model_dump')

    def test_result_defaults(self):
        """Test result has proper defaults."""
        result = LinkToParseResult()
        assert result.node_constraints == []
        assert result.edge_constraints == []
        assert result.special_refs == []
        assert result.errors == []

    def test_result_to_dict(self):
        """Test to_dict method."""
        result = parse_link_to("Task:title")
        d = result.to_dict()
        assert "node_constraints" in d
        assert "edge_constraints" in d
        assert "special_refs" in d
        assert "errors" not in d  # errors not included in to_dict

    def test_result_has_errors_method(self):
        """Test has_errors method."""
        good = parse_link_to("Task:title")
        assert not good.has_errors()

        bad = parse_link_to("Invalid")
        assert bad.has_errors()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
