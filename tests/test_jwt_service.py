"""
Test JWT Service

Tests for JWT token generation and validation.
Run with: pytest tests/test_jwt_service.py -v
"""

import pytest
from services.jwt_service import get_jwt_service, JWTService
import jwt as pyjwt
import time


def test_jwt_service_initialization():
    """Test that JWT service initializes correctly"""
    jwt_service = get_jwt_service()
    assert jwt_service is not None
    assert jwt_service.algorithm == "RS256"
    assert jwt_service.issuer == "https://memory.papr.ai"
    assert jwt_service.audience == "neo4j-graphql"


def test_generate_token():
    """Test JWT token generation"""
    jwt_service = get_jwt_service()

    token = jwt_service.generate_token(
        user_id="test_user_123",
        workspace_id="test_workspace_456"
    )

    assert token is not None
    assert isinstance(token, str)
    assert len(token) > 0

    # Decode without verification to check payload
    decoded = pyjwt.decode(token, options={"verify_signature": False})

    assert decoded["sub"] == "test_user_123"
    assert decoded["user_id"] == "test_user_123"
    assert decoded["workspace_id"] == "test_workspace_456"
    assert decoded["iss"] == "https://memory.papr.ai"
    assert decoded["aud"] == "neo4j-graphql"
    assert "exp" in decoded
    assert "iat" in decoded


def test_generate_token_with_optional_fields():
    """Test JWT generation with all optional fields"""
    jwt_service = get_jwt_service()

    token = jwt_service.generate_token(
        user_id="test_user_123",
        workspace_id="test_workspace_456",
        end_user_id="end_user_789",
        roles=["admin", "developer"],
        expires_in_minutes=30
    )

    decoded = pyjwt.decode(token, options={"verify_signature": False})

    assert decoded["user_id"] == "test_user_123"
    assert decoded["workspace_id"] == "test_workspace_456"
    assert decoded["end_user_id"] == "end_user_789"
    assert decoded["roles"] == ["admin", "developer"]


def test_generate_token_without_workspace():
    """Test JWT generation without workspace_id (personal data)"""
    jwt_service = get_jwt_service()

    token = jwt_service.generate_token(
        user_id="test_user_123",
        workspace_id=None
    )

    decoded = pyjwt.decode(token, options={"verify_signature": False})

    assert decoded["user_id"] == "test_user_123"
    assert "workspace_id" not in decoded  # None values should be removed


def test_verify_token():
    """Test JWT token verification"""
    jwt_service = get_jwt_service()

    # Generate token
    token = jwt_service.generate_token(
        user_id="test_user_123",
        workspace_id="test_workspace_456"
    )

    # Verify token
    payload = jwt_service.verify_token(token)

    assert payload["user_id"] == "test_user_123"
    assert payload["workspace_id"] == "test_workspace_456"
    assert payload["sub"] == "test_user_123"


def test_verify_expired_token():
    """Test that expired tokens are rejected"""
    jwt_service = get_jwt_service()

    # Generate token that expires in 1 second
    token = jwt_service.generate_token(
        user_id="test_user_123",
        expires_in_minutes=0  # Expires immediately
    )

    # Wait a moment
    time.sleep(1)

    # Verification should fail
    with pytest.raises(pyjwt.ExpiredSignatureError):
        jwt_service.verify_token(token)


def test_singleton_pattern():
    """Test that get_jwt_service returns the same instance"""
    service1 = get_jwt_service()
    service2 = get_jwt_service()

    assert service1 is service2


def test_token_structure():
    """Test that token has correct structure (header.payload.signature)"""
    jwt_service = get_jwt_service()

    token = jwt_service.generate_token(user_id="test_user_123")

    # JWT should have 3 parts separated by dots
    parts = token.split('.')
    assert len(parts) == 3

    # Decode header
    header = pyjwt.get_unverified_header(token)
    assert header["alg"] == "RS256"
    assert header["typ"] == "JWT"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
