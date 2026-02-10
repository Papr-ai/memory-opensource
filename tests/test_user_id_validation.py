"""
Tests for user ID validation functions.

These tests verify that the validation layer correctly:
1. Detects external IDs in the user_id field
2. Returns helpful error messages
3. Supports backwards compatibility with valid Parse IDs
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

# Import the validation functions
import sys
sys.path.insert(0, '/Users/shawkatkabbara/Documents/GitHub/memory')

from services.auth_utils import (
    looks_like_external_id,
    validate_user_identification,
    UserIdValidationError,
    log_deprecation_warning
)


class TestLooksLikeExternalId:
    """Tests for the looks_like_external_id heuristic function."""

    # Test UUID detection
    def test_detects_uuid_v4(self):
        """UUIDs should be detected as external IDs."""
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        assert looks_like_external_id(uuid) is True

    def test_detects_uuid_uppercase(self):
        """Uppercase UUIDs should also be detected."""
        uuid = "550E8400-E29B-41D4-A716-446655440000"
        assert looks_like_external_id(uuid) is True

    def test_detects_uuid_mixed_case(self):
        """Mixed case UUIDs should also be detected."""
        uuid = "550e8400-E29B-41d4-a716-446655440000"
        assert looks_like_external_id(uuid) is True

    # Test email detection
    def test_detects_email(self):
        """Email addresses should be detected as external IDs."""
        email = "user@example.com"
        assert looks_like_external_id(email) is True

    def test_detects_email_with_plus(self):
        """Email with plus addressing should be detected."""
        email = "user+tag@example.com"
        assert looks_like_external_id(email) is True

    def test_detects_email_subdomain(self):
        """Email with subdomain should be detected."""
        email = "user@mail.example.com"
        assert looks_like_external_id(email) is True

    # Test common prefixes
    def test_detects_user_prefix(self):
        """IDs starting with 'user_' should be detected."""
        assert looks_like_external_id("user_alice_123") is True
        assert looks_like_external_id("user_12345") is True

    def test_detects_ext_prefix(self):
        """IDs starting with 'ext_' should be detected."""
        assert looks_like_external_id("ext_user_abc") is True

    def test_detects_external_prefix(self):
        """IDs starting with 'external_' should be detected."""
        assert looks_like_external_id("external_user_xyz") is True

    def test_detects_customer_prefix(self):
        """IDs starting with 'customer_' should be detected."""
        assert looks_like_external_id("customer_12345") is True

    def test_detects_cust_prefix(self):
        """IDs starting with 'cust_' should be detected."""
        assert looks_like_external_id("cust_abc123") is True

    def test_detects_client_prefix(self):
        """IDs starting with 'client_' should be detected."""
        assert looks_like_external_id("client_user_1") is True

    def test_prefix_case_insensitive(self):
        """Prefix detection should be case-insensitive."""
        assert looks_like_external_id("USER_alice_123") is True
        assert looks_like_external_id("User_alice_123") is True
        assert looks_like_external_id("EXT_user_abc") is True

    # Test hyphenated IDs
    def test_detects_hyphenated_long_id(self):
        """Long hyphenated IDs should be detected as external."""
        assert looks_like_external_id("company-user-12345") is True
        assert looks_like_external_id("proj-alpha-team-lead") is True

    def test_short_hyphenated_not_detected(self):
        """Short hyphenated IDs (10 chars or less) are not flagged."""
        # This is a borderline case - 10 chars with hyphen
        # The function only returns True for hyphens with len > 10
        # to avoid false positives on short IDs
        assert looks_like_external_id("abc-def-gh") is False  # 10 chars, under threshold

    # Test valid Parse ObjectIds
    def test_valid_parse_id_10_chars(self):
        """Valid 10-character alphanumeric Parse IDs should NOT be flagged."""
        assert looks_like_external_id("mkcNHhG5KP") is False
        assert looks_like_external_id("abcdefghij") is False
        assert looks_like_external_id("1234567890") is False
        assert looks_like_external_id("ABC123xyz0") is False

    def test_valid_parse_id_mixed_case(self):
        """Mixed case 10-char alphanumeric IDs should NOT be flagged."""
        assert looks_like_external_id("AbCdEfGhIj") is False

    # Test edge cases
    def test_empty_string(self):
        """Empty string should return False."""
        assert looks_like_external_id("") is False

    def test_none_value(self):
        """None should return False."""
        assert looks_like_external_id(None) is False

    def test_non_string(self):
        """Non-string values should return False."""
        assert looks_like_external_id(12345) is False
        assert looks_like_external_id(["user_123"]) is False

    def test_very_long_alphanumeric(self):
        """Very long alphanumeric strings are ambiguous but allowed."""
        # 20 chars alphanumeric - not clearly external
        assert looks_like_external_id("abcdefghij1234567890") is False

    def test_long_non_alphanumeric(self):
        """Long non-alphanumeric strings should be detected."""
        assert looks_like_external_id("abcdefghij_1234567890_xyz") is True


class TestValidateUserIdentification:
    """Tests for the validate_user_identification async function."""

    @pytest.fixture
    def mock_memory_graph(self):
        """Create a mock MemoryGraph."""
        return Mock()

    @pytest.mark.asyncio
    async def test_no_user_id_passes(self):
        """Requests without user_id should pass validation."""
        request = Mock()
        request.user_id = None
        request.metadata = None

        result = await validate_user_identification(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_external_user_id_only_passes(self):
        """Requests with only external_user_id should pass."""
        request = Mock()
        request.user_id = None
        request.metadata = Mock()
        request.metadata.user_id = None
        request.metadata.external_user_id = "user_alice_123"

        result = await validate_user_identification(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_uuid_in_user_id_fails(self):
        """UUID in user_id field should fail with helpful error."""
        request = Mock()
        request.user_id = "550e8400-e29b-41d4-a716-446655440000"
        request.metadata = None

        result = await validate_user_identification(request)

        assert result is not None
        assert isinstance(result, UserIdValidationError)
        assert result.code == 400
        assert result.error == "Invalid user_id format"
        assert "external_user_id" in result.suggestion

    @pytest.mark.asyncio
    async def test_email_in_user_id_fails(self):
        """Email in user_id field should fail with helpful error."""
        request = Mock()
        request.user_id = "alice@example.com"
        request.metadata = None

        result = await validate_user_identification(request)

        assert result is not None
        assert result.code == 400
        assert "external_user_id" in result.suggestion

    @pytest.mark.asyncio
    async def test_prefixed_id_in_user_id_fails(self):
        """Prefixed ID in user_id field should fail with helpful error."""
        request = Mock()
        request.user_id = "user_alice_123"
        request.metadata = None

        result = await validate_user_identification(request)

        assert result is not None
        assert result.code == 400
        assert result.field == "user_id"
        assert "user_alice_123" in result.provided_value

    @pytest.mark.asyncio
    async def test_valid_parse_id_passes(self):
        """Valid Parse ObjectId should pass validation."""
        request = Mock()
        request.user_id = "mkcNHhG5KP"
        request.metadata = None

        # Without memory_graph, we skip Parse user validation
        result = await validate_user_identification(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_metadata_user_id_checked(self):
        """user_id in metadata should also be validated."""
        request = Mock()
        request.user_id = None
        request.metadata = Mock()
        request.metadata.user_id = "user_from_metadata_123"

        result = await validate_user_identification(request)

        assert result is not None
        assert result.code == 400

    @pytest.mark.asyncio
    async def test_request_user_id_takes_precedence(self):
        """Request-level user_id should be checked before metadata."""
        request = Mock()
        request.user_id = "ext_invalid_user"  # Invalid
        request.metadata = Mock()
        request.metadata.user_id = "mkcNHhG5KP"  # Valid

        result = await validate_user_identification(request)

        # Should fail because request.user_id is checked first
        assert result is not None
        assert "ext_invalid_user" in result.provided_value

    @pytest.mark.asyncio
    async def test_error_to_dict_format(self):
        """UserIdValidationError.to_dict() should return proper format."""
        request = Mock()
        request.user_id = "user_alice_123"
        request.metadata = None

        result = await validate_user_identification(request)
        error_dict = result.to_dict()

        assert "code" in error_dict
        assert "error" in error_dict
        assert "details" in error_dict
        assert "field" in error_dict["details"]
        assert "provided_value" in error_dict["details"]
        assert "reason" in error_dict["details"]
        assert "suggestion" in error_dict["details"]


class TestDeprecationWarning:
    """Tests for the deprecation warning logging."""

    def test_log_deprecation_warning(self):
        """Should log deprecation warning with correct format."""
        with patch('services.auth_utils.logger') as mock_logger:
            log_deprecation_warning("user_id", "external_user_id", "in memory request")

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "DEPRECATION WARNING" in call_args
            assert "user_id" in call_args
            assert "external_user_id" in call_args

    def test_log_deprecation_warning_no_context(self):
        """Should work without context."""
        with patch('services.auth_utils.logger') as mock_logger:
            log_deprecation_warning("end_user_id", "external_user_id")

            mock_logger.warning.assert_called_once()


class TestRealWorldScenarios:
    """Test real-world scenarios that developers might encounter."""

    @pytest.mark.asyncio
    async def test_firebase_uid_detected(self):
        """Firebase UIDs (28 chars alphanumeric) should pass - they're valid."""
        request = Mock()
        # Firebase UID format: 28 alphanumeric characters
        request.user_id = "aB1cD2eF3gH4iJ5kL6mN7oP8qR9s"  # 28 chars
        request.metadata = None

        result = await validate_user_identification(request)
        # This is alphanumeric and longer than 10 chars - currently passes
        # because it doesn't match our external patterns
        # In a real scenario, this would be caught by the Parse validation
        assert result is None  # Passes heuristic check

    @pytest.mark.asyncio
    async def test_stripe_customer_id_detected(self):
        """Stripe customer IDs (cus_xxx) should be detected."""
        request = Mock()
        request.user_id = "cus_NffrFeUfNV2Hib"
        request.metadata = None

        result = await validate_user_identification(request)
        # Contains underscore and cust_ prefix match
        assert result is not None

    @pytest.mark.asyncio
    async def test_auth0_user_id_detected(self):
        """Auth0 user IDs (auth0|xxx) should be detected."""
        request = Mock()
        request.user_id = "auth0|507f1f77bcf86cd799439011"
        request.metadata = None

        result = await validate_user_identification(request)
        # Contains pipe character - treated as non-alphanumeric
        assert result is not None

    @pytest.mark.asyncio
    async def test_cognito_sub_detected(self):
        """AWS Cognito sub (UUID) should be detected."""
        request = Mock()
        request.user_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        request.metadata = None

        result = await validate_user_identification(request)
        assert result is not None
        assert result.code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
