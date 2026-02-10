"""
Tests for when clause evaluation in memory policies.

Tests the _evaluate_when_condition function which supports:
- Simple property matching
- _and operator (all conditions must match)
- _or operator (at least one must match)
- _not operator (negation)
- Complex nested conditions

Run with: pytest tests/test_memory_policy/test_when_clause_evaluation.py -v
"""

import pytest
from services.node_constraint_resolver import _evaluate_when_condition


class TestSimpleConditions:
    """Test basic property matching without operators."""

    def test_empty_condition_returns_true(self):
        """Empty condition should always pass."""
        result = _evaluate_when_condition(
            condition={},
            node_properties={"severity": "high"}
        )
        assert result is True

    def test_none_condition_returns_true(self):
        """None condition should always pass."""
        result = _evaluate_when_condition(
            condition=None,
            node_properties={"severity": "high"}
        )
        assert result is True

    def test_single_property_match(self):
        """Single property that matches should return True."""
        result = _evaluate_when_condition(
            condition={"severity": "critical"},
            node_properties={"severity": "critical", "status": "active"}
        )
        assert result is True

    def test_single_property_no_match(self):
        """Single property that doesn't match should return False."""
        result = _evaluate_when_condition(
            condition={"severity": "critical"},
            node_properties={"severity": "high", "status": "active"}
        )
        assert result is False

    def test_multiple_properties_all_match(self):
        """Multiple properties that all match should return True."""
        result = _evaluate_when_condition(
            condition={"severity": "critical", "category": "access_control"},
            node_properties={
                "severity": "critical",
                "category": "access_control",
                "status": "active"
            }
        )
        assert result is True

    def test_multiple_properties_partial_match(self):
        """Multiple properties with partial match should return False."""
        result = _evaluate_when_condition(
            condition={"severity": "critical", "category": "access_control"},
            node_properties={
                "severity": "critical",
                "category": "data_protection",
                "status": "active"
            }
        )
        assert result is False

    def test_property_missing_in_node(self):
        """Property not present in node should not match."""
        result = _evaluate_when_condition(
            condition={"severity": "critical"},
            node_properties={"status": "active"}  # No severity
        )
        assert result is False

    def test_none_node_properties(self):
        """None node_properties should be handled gracefully."""
        result = _evaluate_when_condition(
            condition={"severity": "critical"},
            node_properties=None
        )
        assert result is False

    def test_boolean_property_match(self):
        """Boolean property matching should work."""
        result = _evaluate_when_condition(
            condition={"acknowledged": True},
            node_properties={"acknowledged": True, "severity": "high"}
        )
        assert result is True

    def test_boolean_property_no_match(self):
        """Boolean property that doesn't match should return False."""
        result = _evaluate_when_condition(
            condition={"acknowledged": True},
            node_properties={"acknowledged": False, "severity": "high"}
        )
        assert result is False

    def test_integer_property_match(self):
        """Integer property matching should work."""
        result = _evaluate_when_condition(
            condition={"priority_score": 5},
            node_properties={"priority_score": 5, "status": "active"}
        )
        assert result is True


class TestAndOperator:
    """Test _and operator - all conditions must match."""

    def test_and_all_true(self):
        """_and with all true conditions should return True."""
        result = _evaluate_when_condition(
            condition={
                "_and": [
                    {"severity": "critical"},
                    {"category": "access_control"}
                ]
            },
            node_properties={
                "severity": "critical",
                "category": "access_control"
            }
        )
        assert result is True

    def test_and_one_false(self):
        """_and with one false condition should return False."""
        result = _evaluate_when_condition(
            condition={
                "_and": [
                    {"severity": "critical"},
                    {"category": "access_control"}
                ]
            },
            node_properties={
                "severity": "critical",
                "category": "data_protection"  # Different category
            }
        )
        assert result is False

    def test_and_all_false(self):
        """_and with all false conditions should return False."""
        result = _evaluate_when_condition(
            condition={
                "_and": [
                    {"severity": "critical"},
                    {"category": "access_control"}
                ]
            },
            node_properties={
                "severity": "low",
                "category": "data_protection"
            }
        )
        assert result is False

    def test_and_empty_list(self):
        """_and with empty list should return True (vacuous truth)."""
        result = _evaluate_when_condition(
            condition={"_and": []},
            node_properties={"severity": "high"}
        )
        assert result is True

    def test_and_single_item(self):
        """_and with single item should work."""
        result = _evaluate_when_condition(
            condition={"_and": [{"severity": "critical"}]},
            node_properties={"severity": "critical"}
        )
        assert result is True

    def test_and_three_conditions(self):
        """_and with three conditions should work."""
        result = _evaluate_when_condition(
            condition={
                "_and": [
                    {"severity": "critical"},
                    {"category": "access_control"},
                    {"acknowledged": False}
                ]
            },
            node_properties={
                "severity": "critical",
                "category": "access_control",
                "acknowledged": False
            }
        )
        assert result is True


class TestOrOperator:
    """Test _or operator - at least one condition must match."""

    def test_or_first_true(self):
        """_or with first condition true should return True."""
        result = _evaluate_when_condition(
            condition={
                "_or": [
                    {"severity": "critical"},
                    {"severity": "high"}
                ]
            },
            node_properties={"severity": "critical"}
        )
        assert result is True

    def test_or_second_true(self):
        """_or with second condition true should return True."""
        result = _evaluate_when_condition(
            condition={
                "_or": [
                    {"severity": "critical"},
                    {"severity": "high"}
                ]
            },
            node_properties={"severity": "high"}
        )
        assert result is True

    def test_or_all_true(self):
        """_or with all conditions true should return True."""
        result = _evaluate_when_condition(
            condition={
                "_or": [
                    {"status": "active"},
                    {"status": "active"}  # Same condition
                ]
            },
            node_properties={"status": "active"}
        )
        assert result is True

    def test_or_none_true(self):
        """_or with no conditions true should return False."""
        result = _evaluate_when_condition(
            condition={
                "_or": [
                    {"severity": "critical"},
                    {"severity": "high"}
                ]
            },
            node_properties={"severity": "low"}
        )
        assert result is False

    def test_or_empty_list(self):
        """_or with empty list should return False."""
        result = _evaluate_when_condition(
            condition={"_or": []},
            node_properties={"severity": "high"}
        )
        assert result is False

    def test_or_single_item_true(self):
        """_or with single true item should return True."""
        result = _evaluate_when_condition(
            condition={"_or": [{"severity": "critical"}]},
            node_properties={"severity": "critical"}
        )
        assert result is True

    def test_or_single_item_false(self):
        """_or with single false item should return False."""
        result = _evaluate_when_condition(
            condition={"_or": [{"severity": "critical"}]},
            node_properties={"severity": "low"}
        )
        assert result is False


class TestNotOperator:
    """Test _not operator - negation."""

    def test_not_inverts_true_to_false(self):
        """_not should invert true condition to false."""
        result = _evaluate_when_condition(
            condition={"_not": {"status": "completed"}},
            node_properties={"status": "completed"}
        )
        assert result is False

    def test_not_inverts_false_to_true(self):
        """_not should invert false condition to true."""
        result = _evaluate_when_condition(
            condition={"_not": {"status": "completed"}},
            node_properties={"status": "active"}
        )
        assert result is True

    def test_not_with_missing_property(self):
        """_not with missing property should return True (property not equal)."""
        result = _evaluate_when_condition(
            condition={"_not": {"status": "completed"}},
            node_properties={"severity": "high"}  # No status
        )
        assert result is True

    def test_not_with_boolean_true(self):
        """_not should work with boolean True."""
        result = _evaluate_when_condition(
            condition={"_not": {"acknowledged": True}},
            node_properties={"acknowledged": False}
        )
        assert result is True

    def test_not_with_boolean_false(self):
        """_not should work with boolean False."""
        result = _evaluate_when_condition(
            condition={"_not": {"acknowledged": False}},
            node_properties={"acknowledged": False}
        )
        assert result is False


class TestNestedConditions:
    """Test complex nested conditions combining operators."""

    def test_and_containing_or(self):
        """_and containing _or should work correctly."""
        # condition: severity=critical AND (category=access_control OR category=data_protection)
        result = _evaluate_when_condition(
            condition={
                "_and": [
                    {"severity": "critical"},
                    {
                        "_or": [
                            {"category": "access_control"},
                            {"category": "data_protection"}
                        ]
                    }
                ]
            },
            node_properties={
                "severity": "critical",
                "category": "access_control"
            }
        )
        assert result is True

    def test_and_containing_or_fails_on_first(self):
        """_and containing _or should fail if first part fails."""
        result = _evaluate_when_condition(
            condition={
                "_and": [
                    {"severity": "critical"},
                    {
                        "_or": [
                            {"category": "access_control"},
                            {"category": "data_protection"}
                        ]
                    }
                ]
            },
            node_properties={
                "severity": "low",  # Not critical
                "category": "access_control"
            }
        )
        assert result is False

    def test_and_containing_or_fails_on_or(self):
        """_and containing _or should fail if _or part fails."""
        result = _evaluate_when_condition(
            condition={
                "_and": [
                    {"severity": "critical"},
                    {
                        "_or": [
                            {"category": "access_control"},
                            {"category": "data_protection"}
                        ]
                    }
                ]
            },
            node_properties={
                "severity": "critical",
                "category": "audit"  # Neither access_control nor data_protection
            }
        )
        assert result is False

    def test_or_containing_and(self):
        """_or containing _and should work correctly."""
        # condition: (severity=critical AND urgent=true) OR (status=emergency)
        result = _evaluate_when_condition(
            condition={
                "_or": [
                    {
                        "_and": [
                            {"severity": "critical"},
                            {"urgent": True}
                        ]
                    },
                    {"status": "emergency"}
                ]
            },
            node_properties={
                "severity": "critical",
                "urgent": True
            }
        )
        assert result is True

    def test_or_containing_and_second_branch(self):
        """_or containing _and should match on second branch."""
        result = _evaluate_when_condition(
            condition={
                "_or": [
                    {
                        "_and": [
                            {"severity": "critical"},
                            {"urgent": True}
                        ]
                    },
                    {"status": "emergency"}
                ]
            },
            node_properties={
                "status": "emergency"  # Second branch
            }
        )
        assert result is True

    def test_and_containing_not(self):
        """_and containing _not should work correctly."""
        # condition: severity=critical AND NOT(acknowledged=true)
        result = _evaluate_when_condition(
            condition={
                "_and": [
                    {"severity": "critical"},
                    {"_not": {"acknowledged": True}}
                ]
            },
            node_properties={
                "severity": "critical",
                "acknowledged": False
            }
        )
        assert result is True

    def test_and_containing_not_fails(self):
        """_and containing _not should fail if _not fails."""
        result = _evaluate_when_condition(
            condition={
                "_and": [
                    {"severity": "critical"},
                    {"_not": {"acknowledged": True}}
                ]
            },
            node_properties={
                "severity": "critical",
                "acknowledged": True  # Already acknowledged
            }
        )
        assert result is False

    def test_deeply_nested_conditions(self):
        """Deeply nested conditions should work correctly."""
        # condition: severity=critical AND ((category=access_control OR category=data_protection) AND NOT(acknowledged=true))
        result = _evaluate_when_condition(
            condition={
                "_and": [
                    {"severity": "critical"},
                    {
                        "_and": [
                            {
                                "_or": [
                                    {"category": "access_control"},
                                    {"category": "data_protection"}
                                ]
                            },
                            {"_not": {"acknowledged": True}}
                        ]
                    }
                ]
            },
            node_properties={
                "severity": "critical",
                "category": "access_control",
                "acknowledged": False
            }
        )
        assert result is True

    def test_deeptrust_security_condition(self):
        """Real-world DeepTrust condition: critical tactic not yet acknowledged."""
        # From DeepTrust example: Mark critical tactics that haven't been acknowledged
        result = _evaluate_when_condition(
            condition={
                "_and": [
                    {"severity": "critical"},
                    {
                        "_or": [
                            {"category": "access_control"},
                            {"category": "data_protection"}
                        ]
                    },
                    {"_not": {"acknowledged": True}}
                ]
            },
            node_properties={
                "severity": "critical",
                "category": "access_control",
                "acknowledged": False,
                "tactic_name": "Defense Evasion"
            }
        )
        assert result is True


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_node_properties(self):
        """Empty node properties should handle gracefully."""
        result = _evaluate_when_condition(
            condition={"severity": "critical"},
            node_properties={}
        )
        assert result is False

    def test_condition_with_unknown_operator(self):
        """Condition with unknown operator key starting with _ should be skipped."""
        # The implementation skips keys starting with _ that aren't recognized operators
        result = _evaluate_when_condition(
            condition={
                "_unknown_operator": [{"severity": "critical"}],
                "status": "active"  # This should still be evaluated
            },
            node_properties={"status": "active"}
        )
        assert result is True

    def test_nested_none_properties(self):
        """None value in properties should be handled."""
        result = _evaluate_when_condition(
            condition={"severity": None},
            node_properties={"severity": None}
        )
        assert result is True

    def test_none_value_no_match(self):
        """None value that doesn't match should return False."""
        result = _evaluate_when_condition(
            condition={"severity": None},
            node_properties={"severity": "high"}
        )
        assert result is False

    def test_with_context_parameter(self):
        """Context parameter should be passed through (even if unused in simple conditions)."""
        result = _evaluate_when_condition(
            condition={"severity": "critical"},
            node_properties={"severity": "critical"},
            context={"user_id": "test_user", "workspace_id": "ws_123"}
        )
        assert result is True

    def test_list_value_match(self):
        """List value matching should work."""
        result = _evaluate_when_condition(
            condition={"tags": ["urgent", "bug"]},
            node_properties={"tags": ["urgent", "bug"]}
        )
        assert result is True

    def test_list_value_no_match(self):
        """List value that doesn't match should return False."""
        result = _evaluate_when_condition(
            condition={"tags": ["urgent", "bug"]},
            node_properties={"tags": ["urgent", "feature"]}
        )
        assert result is False

    def test_dict_value_match(self):
        """Dict value matching should work."""
        result = _evaluate_when_condition(
            condition={"metadata": {"priority": "high"}},
            node_properties={"metadata": {"priority": "high"}}
        )
        assert result is True


class TestDocumentedExamples:
    """Test examples from the DX documentation."""

    def test_simple_condition_from_docs(self):
        """Test: link_to={"Task:title": {"when": {"priority": "high"}}}"""
        result = _evaluate_when_condition(
            condition={"priority": "high"},
            node_properties={"priority": "high", "status": "active"}
        )
        assert result is True

    def test_and_condition_from_docs(self):
        """Test: {"_and": [{"priority": "high"}, {"status": "active"}]}"""
        result = _evaluate_when_condition(
            condition={
                "_and": [
                    {"priority": "high"},
                    {"status": "active"}
                ]
            },
            node_properties={"priority": "high", "status": "active"}
        )
        assert result is True

    def test_or_condition_from_docs(self):
        """Test: {"_or": [{"status": "active"}, {"status": "pending"}]}"""
        result = _evaluate_when_condition(
            condition={
                "_or": [
                    {"status": "active"},
                    {"status": "pending"}
                ]
            },
            node_properties={"status": "pending"}
        )
        assert result is True

    def test_not_condition_from_docs(self):
        """Test: {"_not": {"status": "completed"}}"""
        result = _evaluate_when_condition(
            condition={"_not": {"status": "completed"}},
            node_properties={"status": "in_progress"}
        )
        assert result is True

    def test_complex_condition_from_docs(self):
        """Test: priority=high AND NOT completed"""
        result = _evaluate_when_condition(
            condition={
                "_and": [
                    {"priority": "high"},
                    {"_not": {"status": "completed"}}
                ]
            },
            node_properties={"priority": "high", "status": "active"}
        )
        assert result is True

    def test_security_monitoring_condition(self):
        """Test condition from security monitoring example."""
        # From docs: Alert on violations with critical or high severity
        result = _evaluate_when_condition(
            condition={
                "_or": [
                    {"severity": "critical"},
                    {"severity": "high"}
                ]
            },
            node_properties={
                "severity": "critical",
                "type": "SQL injection"
            }
        )
        assert result is True
