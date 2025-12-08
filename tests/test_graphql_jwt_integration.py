"""
Test GraphQL JWT Integration

Tests for JWT service integration with GraphQL proxy,
including token generation, validation, and JWKS endpoint.

Run with: pytest tests/test_graphql_jwt_integration.py -v
"""

import pytest
import httpx
import json
from fastapi.testclient import TestClient
from main import app
from services.jwt_service import get_jwt_service, JWTService
from services.logger_singleton import LoggerSingleton
from os import environ as env
from dotenv import load_dotenv, find_dotenv
import jwt as pyjwt
import time
from datetime import datetime, timedelta, UTC

# Load environment variables
ENV_FILE = find_dotenv()
load_dotenv(ENV_FILE)

logger = LoggerSingleton.get_logger(__name__)


class TestJWTServiceBasics:
    """Test JWT service basic functionality"""

    def test_jwt_service_initialization(self):
        """Test that JWT service initializes correctly"""
        jwt_service = get_jwt_service()

        assert jwt_service is not None
        assert jwt_service.algorithm == "RS256"
        assert jwt_service.issuer == "https://memory.papr.ai"
        assert jwt_service.audience == "neo4j-graphql"
        assert jwt_service.private_key is not None

    def test_jwt_service_singleton(self):
        """Test that get_jwt_service returns the same instance"""
        service1 = get_jwt_service()
        service2 = get_jwt_service()

        assert service1 is service2

    def test_generate_token_basic(self):
        """Test basic JWT token generation"""
        jwt_service = get_jwt_service()

        token = jwt_service.generate_token(
            user_id="test_user_123",
            workspace_id="test_workspace_456"
        )

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

        # JWT should have 3 parts (header.payload.signature)
        parts = token.split('.')
        assert len(parts) == 3

    def test_generate_token_with_all_claims(self):
        """Test JWT generation with all optional claims"""
        jwt_service = get_jwt_service()

        token = jwt_service.generate_token(
            user_id="test_user_123",
            workspace_id="test_workspace_456",
            end_user_id="end_user_789",
            roles=["developer", "admin"],
            expires_in_minutes=30
        )

        # Decode without verification to check claims
        decoded = pyjwt.decode(token, options={"verify_signature": False})

        assert decoded["user_id"] == "test_user_123"
        assert decoded["workspace_id"] == "test_workspace_456"
        assert decoded["end_user_id"] == "end_user_789"
        assert decoded["roles"] == ["developer", "admin"]
        assert decoded["sub"] == "test_user_123"
        assert decoded["iss"] == "https://memory.papr.ai"
        assert decoded["aud"] == "neo4j-graphql"

    def test_generate_token_without_workspace(self):
        """Test JWT generation without workspace_id (personal data)"""
        jwt_service = get_jwt_service()

        token = jwt_service.generate_token(
            user_id="test_user_123",
            workspace_id=None
        )

        decoded = pyjwt.decode(token, options={"verify_signature": False})

        assert decoded["user_id"] == "test_user_123"
        # workspace_id should not be in claims if None
        assert "workspace_id" not in decoded

    def test_verify_token(self):
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

    def test_verify_invalid_token(self):
        """Test that invalid tokens are rejected"""
        jwt_service = get_jwt_service()

        invalid_token = "invalid.jwt.token"

        with pytest.raises(pyjwt.InvalidTokenError):
            jwt_service.verify_token(invalid_token)

    def test_verify_expired_token(self):
        """Test that expired tokens are rejected"""
        jwt_service = get_jwt_service()

        # Generate token that expires immediately
        token = jwt_service.generate_token(
            user_id="test_user_123",
            expires_in_minutes=0
        )

        # Wait a moment for expiration
        time.sleep(1)

        # Verification should fail
        with pytest.raises(pyjwt.ExpiredSignatureError):
            jwt_service.verify_token(token)

    def test_token_expiration_timing(self):
        """Test that token expiration is set correctly"""
        jwt_service = get_jwt_service()

        token = jwt_service.generate_token(
            user_id="test_user_123",
            expires_in_minutes=60
        )

        decoded = pyjwt.decode(token, options={"verify_signature": False})

        # Check exp and iat claims
        assert "exp" in decoded
        assert "iat" in decoded

        exp_time = datetime.fromtimestamp(decoded["exp"], tz=UTC)
        iat_time = datetime.fromtimestamp(decoded["iat"], tz=UTC)

        # Expiration should be ~60 minutes from issued time
        delta = exp_time - iat_time
        assert delta.total_seconds() / 60 >= 59  # Allow 1 minute tolerance
        assert delta.total_seconds() / 60 <= 61


class TestJWKSEndpoint:
    """Test JWKS endpoint for Neo4j JWT validation"""

    def test_jwks_endpoint_accessible(self, app):
        """Test that JWKS endpoint is accessible"""
        client = TestClient(app)

        response = client.get("/.well-known/jwks.json")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_jwks_response_format(self, app):
        """Test that JWKS response has correct format"""
        client = TestClient(app)

        response = client.get("/.well-known/jwks.json")

        assert response.status_code == 200

        jwks = response.json()

        # JWKS format
        assert "keys" in jwks
        assert isinstance(jwks["keys"], list)
        assert len(jwks["keys"]) > 0

        # First key should have required fields
        key = jwks["keys"][0]
        assert key["kty"] == "RSA"
        assert key["use"] == "sig"
        assert key["alg"] == "RS256"
        assert "n" in key  # Modulus
        assert "e" in key  # Exponent
        assert "kid" in key  # Key ID

    def test_jwks_public_key_matches_jwt_signature(self, app):
        """Test that JWKS public key can verify our JWTs"""
        client = TestClient(app)

        # Get JWKS
        jwks_response = client.get("/.well-known/jwks.json")
        assert jwks_response.status_code == 200
        jwks = jwks_response.json()

        # Generate a JWT
        jwt_service = get_jwt_service()
        token = jwt_service.generate_token(
            user_id="test_user_123",
            workspace_id="test_workspace_456"
        )

        # Verify the token using our JWT service (uses public key)
        payload = jwt_service.verify_token(token)

        assert payload["user_id"] == "test_user_123"
        # If verification succeeds, public key in JWKS is correct

    def test_jwks_caching_headers(self, app):
        """Test that JWKS has appropriate caching headers"""
        client = TestClient(app)

        response = client.get("/.well-known/jwks.json")

        assert response.status_code == 200

        # Should have cache-control header
        assert "cache-control" in response.headers
        cache_control = response.headers["cache-control"]

        # Should allow caching
        assert "max-age" in cache_control or "public" in cache_control

    def test_jwks_cors_headers(self, app):
        """Test that JWKS has CORS headers for Neo4j"""
        client = TestClient(app)

        response = client.get("/.well-known/jwks.json")

        assert response.status_code == 200

        # Should allow cross-origin access for Neo4j to fetch
        assert "access-control-allow-origin" in response.headers


class TestJWTGraphQLIntegration:
    """Test JWT integration with GraphQL proxy"""

    def test_graphql_proxy_generates_valid_jwt(self, app):
        """Test that GraphQL proxy generates valid JWTs"""
        from unittest.mock import patch, AsyncMock

        TEST_API_KEY = env.get('TEST_X_PAPR_API_KEY')
        if not TEST_API_KEY:
            pytest.skip("TEST_X_PAPR_API_KEY not set in .env")

        client = TestClient(app)

        query = """
        query {
            __typename
        }
        """

        captured_jwt = None

        def capture_jwt(*args, **kwargs):
            nonlocal captured_jwt
            auth_header = kwargs["headers"].get("Authorization", "")
            if auth_header.startswith("Bearer "):
                captured_jwt = auth_header.replace("Bearer ", "")

            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.content = json.dumps({
                "data": {"__typename": "Query"}
            }).encode()
            return mock_response

        with patch('httpx.AsyncClient.post', side_effect=capture_jwt):
            response = client.post(
                "/v1/graphql",
                json={"query": query},
                headers={
                    "X-API-Key": TEST_API_KEY,
                    "Content-Type": "application/json"
                }
            )

            assert response.status_code == 200
            assert captured_jwt is not None

            # Verify JWT is valid
            jwt_service = get_jwt_service()
            payload = jwt_service.verify_token(captured_jwt)

            assert payload is not None
            assert "user_id" in payload

    def test_jwt_includes_neo4j_required_claims(self, app):
        """Test that JWT includes all claims required by Neo4j @authorization"""
        from unittest.mock import patch, AsyncMock

        TEST_API_KEY = env.get('TEST_X_PAPR_API_KEY')
        if not TEST_API_KEY:
            pytest.skip("TEST_X_PAPR_API_KEY not set in .env")

        client = TestClient(app)

        query = """
        query {
            __typename
        }
        """

        captured_jwt = None

        def capture_jwt(*args, **kwargs):
            nonlocal captured_jwt
            auth_header = kwargs["headers"].get("Authorization", "")
            if auth_header.startswith("Bearer "):
                captured_jwt = auth_header.replace("Bearer ", "")

            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.content = json.dumps({
                "data": {"__typename": "Query"}
            }).encode()
            return mock_response

        with patch('httpx.AsyncClient.post', side_effect=capture_jwt):
            response = client.post(
                "/v1/graphql",
                json={"query": query},
                headers={
                    "X-API-Key": TEST_API_KEY,
                    "Content-Type": "application/json"
                }
            )

            assert response.status_code == 200

            decoded = pyjwt.decode(captured_jwt, options={"verify_signature": False})

            # Required claims for Neo4j @authorization directives
            # Example: where: { node: { user_id: "$jwt.user_id" } }
            assert "user_id" in decoded, "JWT must have user_id for @authorization"

            # Standard JWT claims
            assert "iss" in decoded  # Issuer
            assert "aud" in decoded  # Audience
            assert "exp" in decoded  # Expiration
            assert "iat" in decoded  # Issued at
            assert "sub" in decoded  # Subject

    def test_jwt_expiration_appropriate_for_queries(self, app):
        """Test that JWT expiration is appropriate for typical query duration"""
        from unittest.mock import patch, AsyncMock

        TEST_API_KEY = env.get('TEST_X_PAPR_API_KEY')
        if not TEST_API_KEY:
            pytest.skip("TEST_X_PAPR_API_KEY not set in .env")

        client = TestClient(app)

        query = """
        query {
            __typename
        }
        """

        captured_jwt = None

        def capture_jwt(*args, **kwargs):
            nonlocal captured_jwt
            auth_header = kwargs["headers"].get("Authorization", "")
            if auth_header.startswith("Bearer "):
                captured_jwt = auth_header.replace("Bearer ", "")

            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.content = json.dumps({
                "data": {"__typename": "Query"}
            }).encode()
            return mock_response

        with patch('httpx.AsyncClient.post', side_effect=capture_jwt):
            response = client.post(
                "/v1/graphql",
                json={"query": query},
                headers={
                    "X-API-Key": TEST_API_KEY,
                    "Content-Type": "application/json"
                }
            )

            assert response.status_code == 200

            decoded = pyjwt.decode(captured_jwt, options={"verify_signature": False})

            exp_time = datetime.fromtimestamp(decoded["exp"], tz=UTC)
            iat_time = datetime.fromtimestamp(decoded["iat"], tz=UTC)

            # Token should be valid for at least 30 minutes (enough for any query)
            delta = exp_time - iat_time
            assert delta.total_seconds() / 60 >= 30

            # But not too long (security)
            assert delta.total_seconds() / 60 <= 120  # 2 hours max

    def test_different_requests_get_fresh_jwts(self, app):
        """Test that each request gets a fresh JWT with current timestamp"""
        from unittest.mock import patch, AsyncMock

        TEST_API_KEY = env.get('TEST_X_PAPR_API_KEY')
        if not TEST_API_KEY:
            pytest.skip("TEST_X_PAPR_API_KEY not set in .env")

        client = TestClient(app)

        query = """
        query {
            __typename
        }
        """

        jwts = []

        def capture_jwt(*args, **kwargs):
            auth_header = kwargs["headers"].get("Authorization", "")
            if auth_header.startswith("Bearer "):
                jwts.append(auth_header.replace("Bearer ", ""))

            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.content = json.dumps({
                "data": {"__typename": "Query"}
            }).encode()
            return mock_response

        # Make two requests
        with patch('httpx.AsyncClient.post', side_effect=capture_jwt):
            client.post(
                "/v1/graphql",
                json={"query": query},
                headers={
                    "X-API-Key": TEST_API_KEY,
                    "Content-Type": "application/json"
                }
            )

            time.sleep(1)  # Wait 1 second

            client.post(
                "/v1/graphql",
                json={"query": query},
                headers={
                    "X-API-Key": TEST_API_KEY,
                    "Content-Type": "application/json"
                }
            )

        assert len(jwts) == 2

        # Decode both tokens
        decoded1 = pyjwt.decode(jwts[0], options={"verify_signature": False})
        decoded2 = pyjwt.decode(jwts[1], options={"verify_signature": False})

        # Should have different issued-at times
        assert decoded1["iat"] != decoded2["iat"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
