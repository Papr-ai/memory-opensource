"""
Tests for OMO (Open Memory Object) Safety Standards implementation.

These tests verify that the OMO safety pipeline correctly:
1. Enforces consent standards (skip extraction for consent='none')
2. Applies risk-based restrictions
3. Propagates ACL from memory to nodes
4. Creates audit trails for compliance
"""

import pytest
import asyncio
from typing import Dict, List, Any

# Import the OMO safety functions
import sys
sys.path.insert(0, '/Users/shawkatkabbara/Documents/GitHub/memory')

from services.omo_safety import (
    enforce_consent_standard,
    enforce_risk_standard,
    propagate_acl,
    create_audit_trail,
    process_memory_with_omo,
    validate_consent_level,
    validate_risk_level,
    get_extraction_method_from_policy_mode
)


# ============================================================================
# Test Data
# ============================================================================

def create_test_nodes() -> List[Dict[str, Any]]:
    """Create sample nodes for testing."""
    return [
        {
            "label": "Person",
            "properties": {"name": "Alice", "role": "Manager"}
        },
        {
            "label": "Task",
            "properties": {"title": "Review PR", "status": "pending"}
        }
    ]


# ============================================================================
# Consent Enforcement Tests
# ============================================================================

class TestConsentEnforcement:
    """Tests for consent standard enforcement."""

    @pytest.mark.asyncio
    async def test_consent_none_returns_empty(self):
        """consent='none' should return empty list (no nodes created)."""
        nodes = create_test_nodes()
        result = await enforce_consent_standard(
            memory_content="Test content",
            memory_id="mem_123",
            consent="none",
            nodes=nodes
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_consent_explicit_annotates_nodes(self):
        """consent='explicit' should annotate nodes with consent info."""
        nodes = create_test_nodes()
        result = await enforce_consent_standard(
            memory_content="Test content",
            memory_id="mem_123",
            consent="explicit",
            nodes=nodes
        )
        assert len(result) == 2
        assert result[0]["properties"]["_omo_consent"] == "explicit"
        assert result[0]["properties"]["_omo_source_memory_id"] == "mem_123"

    @pytest.mark.asyncio
    async def test_consent_implicit_annotates_nodes(self):
        """consent='implicit' should annotate nodes."""
        nodes = create_test_nodes()
        result = await enforce_consent_standard(
            memory_content="Test",
            memory_id="mem_456",
            consent="implicit",
            nodes=nodes
        )
        assert len(result) == 2
        assert result[0]["properties"]["_omo_consent"] == "implicit"

    @pytest.mark.asyncio
    async def test_consent_terms_annotates_nodes(self):
        """consent='terms' should annotate nodes."""
        nodes = create_test_nodes()
        result = await enforce_consent_standard(
            memory_content="Test",
            memory_id="mem_789",
            consent="terms",
            nodes=nodes
        )
        assert len(result) == 2
        assert result[0]["properties"]["_omo_consent"] == "terms"


# ============================================================================
# Risk Assessment Tests
# ============================================================================

class TestRiskEnforcement:
    """Tests for risk standard enforcement."""

    @pytest.mark.asyncio
    async def test_risk_flagged_restricts_acl(self):
        """risk='flagged' should restrict ACL to owner only."""
        nodes = create_test_nodes()
        result = await enforce_risk_standard(
            memory_id="mem_123",
            risk="flagged",
            external_user_id="user_alice",
            nodes=nodes
        )
        assert len(result) == 2
        assert result[0]["properties"]["_omo_risk"] == "flagged"
        assert result[0]["properties"]["_omo_requires_review"] is True
        assert result[0]["acl"] == {"read": ["user_alice"], "write": ["user_alice"]}

    @pytest.mark.asyncio
    async def test_risk_sensitive_marks_nodes(self):
        """risk='sensitive' should mark nodes but not restrict ACL."""
        nodes = create_test_nodes()
        result = await enforce_risk_standard(
            memory_id="mem_123",
            risk="sensitive",
            external_user_id="user_alice",
            nodes=nodes
        )
        assert len(result) == 2
        assert result[0]["properties"]["_omo_risk"] == "sensitive"
        assert "acl" not in result[0]  # No ACL restriction for sensitive

    @pytest.mark.asyncio
    async def test_risk_none_marks_as_safe(self):
        """risk='none' should mark nodes as safe."""
        nodes = create_test_nodes()
        result = await enforce_risk_standard(
            memory_id="mem_123",
            risk="none",
            external_user_id="user_alice",
            nodes=nodes
        )
        assert len(result) == 2
        assert result[0]["properties"]["_omo_risk"] == "none"


# ============================================================================
# ACL Propagation Tests
# ============================================================================

class TestACLPropagation:
    """Tests for ACL propagation from memory to nodes."""

    @pytest.mark.asyncio
    async def test_explicit_acl_used(self):
        """Explicit omo_acl should be used when provided."""
        nodes = create_test_nodes()
        explicit_acl = {"read": ["user_a", "user_b"], "write": ["user_a"]}
        result = await propagate_acl(
            memory_id="mem_123",
            omo_acl=explicit_acl,
            external_user_id="user_c",
            developer_user_id="dev_123",
            nodes=nodes
        )
        assert result[0]["acl"] == explicit_acl
        assert result[1]["acl"] == explicit_acl

    @pytest.mark.asyncio
    async def test_default_acl_created(self):
        """Default ACL should be created when omo_acl is None."""
        nodes = create_test_nodes()
        result = await propagate_acl(
            memory_id="mem_123",
            omo_acl=None,
            external_user_id="user_alice",
            developer_user_id="dev_123",
            nodes=nodes
        )
        # Both external_user_id and developer_user_id should be in ACL
        assert "user_alice" in result[0]["acl"]["read"]
        assert "dev_123" in result[0]["acl"]["read"]
        assert "user_alice" in result[0]["acl"]["write"]
        assert "dev_123" in result[0]["acl"]["write"]

    @pytest.mark.asyncio
    async def test_skips_nodes_with_existing_acl(self):
        """Should skip nodes that already have ACL set."""
        nodes = [
            {"label": "Task", "properties": {}, "acl": {"read": ["owner"], "write": ["owner"]}}
        ]
        result = await propagate_acl(
            memory_id="mem_123",
            omo_acl={"read": ["override"], "write": ["override"]},
            external_user_id="user_alice",
            developer_user_id="dev_123",
            nodes=nodes
        )
        # ACL should not be overridden
        assert result[0]["acl"] == {"read": ["owner"], "write": ["owner"]}


# ============================================================================
# Audit Trail Tests
# ============================================================================

class TestAuditTrail:
    """Tests for audit trail creation."""

    @pytest.mark.asyncio
    async def test_audit_trail_created(self):
        """Should create audit trail with all required fields."""
        nodes = create_test_nodes()
        result = await create_audit_trail(
            memory_id="mem_123",
            consent="explicit",
            risk="sensitive",
            extraction_method="llm",
            nodes=nodes
        )
        audit = result[0]["properties"]["_omo_audit"]
        assert audit["source_memory_id"] == "mem_123"
        assert audit["consent"] == "explicit"
        assert audit["risk"] == "sensitive"
        assert audit["extraction_method"] == "llm"
        assert "extracted_at" in audit

    @pytest.mark.asyncio
    async def test_audit_trail_manual_extraction(self):
        """Should correctly record manual extraction method."""
        nodes = create_test_nodes()
        result = await create_audit_trail(
            memory_id="mem_123",
            consent="terms",
            risk="none",
            extraction_method="manual",
            nodes=nodes
        )
        audit = result[0]["properties"]["_omo_audit"]
        assert audit["extraction_method"] == "manual"


# ============================================================================
# Full Pipeline Tests
# ============================================================================

class TestOMOPipeline:
    """Tests for the complete OMO safety pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_consent_none(self):
        """Full pipeline should return empty for consent='none'."""
        nodes = create_test_nodes()
        result = await process_memory_with_omo(
            memory_id="mem_123",
            memory_content="Sensitive data",
            extracted_nodes=nodes,
            consent="none",
            risk="none"
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_full_pipeline_explicit_consent(self):
        """Full pipeline should process nodes with explicit consent."""
        nodes = create_test_nodes()
        result = await process_memory_with_omo(
            memory_id="mem_123",
            memory_content="Test content",
            extracted_nodes=nodes,
            consent="explicit",
            risk="none",
            external_user_id="user_alice",
            developer_user_id="dev_123"
        )
        assert len(result) == 2
        # Check all OMO annotations are present
        assert result[0]["properties"]["_omo_consent"] == "explicit"
        assert result[0]["properties"]["_omo_risk"] == "none"
        assert "_omo_audit" in result[0]["properties"]
        assert "acl" in result[0]

    @pytest.mark.asyncio
    async def test_full_pipeline_flagged_risk(self):
        """Full pipeline should restrict ACL for flagged content."""
        nodes = create_test_nodes()
        result = await process_memory_with_omo(
            memory_id="mem_123",
            memory_content="PII content",
            extracted_nodes=nodes,
            consent="explicit",
            risk="flagged",
            external_user_id="user_owner"
        )
        assert len(result) == 2
        assert result[0]["acl"] == {"read": ["user_owner"], "write": ["user_owner"]}
        assert result[0]["properties"]["_omo_requires_review"] is True

    @pytest.mark.asyncio
    async def test_full_pipeline_with_explicit_acl(self):
        """Full pipeline should use explicit ACL when provided."""
        nodes = create_test_nodes()
        explicit_acl = {"read": ["user_a", "user_b"], "write": ["user_a"]}
        result = await process_memory_with_omo(
            memory_id="mem_123",
            memory_content="Shared content",
            extracted_nodes=nodes,
            consent="implicit",
            risk="none",
            omo_acl=explicit_acl
        )
        assert result[0]["acl"] == explicit_acl


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_validate_consent_level_valid(self):
        """Should return True for valid consent levels."""
        assert validate_consent_level("explicit") is True
        assert validate_consent_level("implicit") is True
        assert validate_consent_level("terms") is True
        assert validate_consent_level("none") is True

    def test_validate_consent_level_invalid(self):
        """Should return False for invalid consent levels."""
        assert validate_consent_level("invalid") is False
        assert validate_consent_level("") is False
        assert validate_consent_level("EXPLICIT") is False  # Case-sensitive

    def test_validate_risk_level_valid(self):
        """Should return True for valid risk levels."""
        assert validate_risk_level("none") is True
        assert validate_risk_level("sensitive") is True
        assert validate_risk_level("flagged") is True

    def test_validate_risk_level_invalid(self):
        """Should return False for invalid risk levels."""
        assert validate_risk_level("high") is False
        assert validate_risk_level("") is False
        assert validate_risk_level("NONE") is False  # Case-sensitive

    def test_extraction_method_from_policy_mode(self):
        """Should convert PolicyMode to extraction method."""
        assert get_extraction_method_from_policy_mode("auto") == "llm"
        assert get_extraction_method_from_policy_mode("hybrid") == "llm"
        assert get_extraction_method_from_policy_mode("structured") == "manual"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
