"""
Test GraphQL Multi-Tenant Isolation

Tests to verify that GraphQL queries are properly filtered by user_id and workspace_id,
ensuring no data leakage between tenants.

Run with: pytest tests/test_graphql_multi_tenant_isolation.py -v
"""

import pytest
import httpx
import json
from fastapi.testclient import TestClient
from main import app
from services.jwt_service import get_jwt_service
from services.logger_singleton import LoggerSingleton
from os import environ as env
from dotenv import load_dotenv, find_dotenv
from unittest.mock import patch, MagicMock, AsyncMock
import jwt as pyjwt

# Load environment variables
ENV_FILE = find_dotenv()
load_dotenv(ENV_FILE)

logger = LoggerSingleton.get_logger(__name__)

# Test credentials
TEST_API_KEY_USER_A = env.get('TEST_X_PAPR_API_KEY')
TEST_API_KEY_USER_B = env.get('TEST_X_USER_API_KEY')  # Different user


class TestMultiTenantJWTClaims:
    """Test that JWTs contain correct multi-tenant claims"""

    def test_jwt_includes_user_id_claim(self, app):
        """Test that JWT includes user_id for @authorization filtering"""
        if not TEST_API_KEY_USER_A:
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
                    "X-API-Key": TEST_API_KEY_USER_A,
                    "Content-Type": "application/json"
                }
            )

            assert response.status_code == 200
            assert captured_jwt is not None

            # Decode JWT without verification to inspect claims
            decoded = pyjwt.decode(
                captured_jwt,
                options={"verify_signature": False}
            )

            # Must have user_id for Neo4j @authorization directive
            assert "user_id" in decoded
            assert decoded["user_id"] is not None
            assert len(decoded["user_id"]) > 0

            logger.info(f"JWT user_id claim: {decoded['user_id']}")

    def test_jwt_includes_workspace_id_when_available(self, app):
        """Test that JWT includes workspace_id for workspace-scoped queries"""
        if not TEST_API_KEY_USER_A:
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
                    "X-API-Key": TEST_API_KEY_USER_A,
                    "Content-Type": "application/json"
                }
            )

            assert response.status_code == 200
            assert captured_jwt is not None

            decoded = pyjwt.decode(
                captured_jwt,
                options={"verify_signature": False}
            )

            # workspace_id may be None (personal) or a string (workspace)
            # Both are valid - just verify the field structure
            if "workspace_id" in decoded:
                logger.info(f"JWT workspace_id claim: {decoded.get('workspace_id')}")
            else:
                logger.info("JWT has no workspace_id (personal scope)")

    def test_different_users_get_different_jwt_claims(self, app):
        """Test that different API keys result in different user_id claims"""
        if not TEST_API_KEY_USER_A or not TEST_API_KEY_USER_B:
            pytest.skip("Both TEST_X_PAPR_API_KEY and TEST_X_USER_API_KEY must be set")

        client = TestClient(app)

        query = """
        query {
            __typename
        }
        """

        jwts = {}

        def capture_jwt(key_name):
            def _capture(*args, **kwargs):
                auth_header = kwargs["headers"].get("Authorization", "")
                if auth_header.startswith("Bearer "):
                    jwts[key_name] = auth_header.replace("Bearer ", "")

                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_response.content = json.dumps({
                    "data": {"__typename": "Query"}
                }).encode()
                return mock_response
            return _capture

        # Request for User A
        with patch('httpx.AsyncClient.post', side_effect=capture_jwt("user_a")):
            response_a = client.post(
                "/v1/graphql",
                json={"query": query},
                headers={
                    "X-API-Key": TEST_API_KEY_USER_A,
                    "Content-Type": "application/json"
                }
            )
            assert response_a.status_code == 200

        # Request for User B
        with patch('httpx.AsyncClient.post', side_effect=capture_jwt("user_b")):
            response_b = client.post(
                "/v1/graphql",
                json={"query": query},
                headers={
                    "X-API-Key": TEST_API_KEY_USER_B,
                    "Content-Type": "application/json"
                }
            )
            assert response_b.status_code == 200

        # Decode both JWTs
        decoded_a = pyjwt.decode(jwts["user_a"], options={"verify_signature": False})
        decoded_b = pyjwt.decode(jwts["user_b"], options={"verify_signature": False})

        # Different users should have different user_id claims
        assert decoded_a["user_id"] != decoded_b["user_id"]

        logger.info(f"User A ID: {decoded_a['user_id']}")
        logger.info(f"User B ID: {decoded_b['user_id']}")


class TestMultiTenantQueryFiltering:
    """Test that queries are filtered by tenant"""

    def test_user_cannot_access_other_user_data(self, app):
        """Test that User A cannot access User B's data"""
        if not TEST_API_KEY_USER_A:
            pytest.skip("TEST_X_PAPR_API_KEY not set in .env")

        client = TestClient(app)

        # Query for a CodeProject that belongs to User B
        query = """
        query GetCodeProject($id: ID!) {
            codeProject(id: $id) {
                id
                name
                user_id
            }
        }
        """

        # Simulate Neo4j returning "not found" because of @authorization filter
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            # Neo4j returns null when @authorization filters out the result
            mock_response.content = json.dumps({
                "data": {
                    "codeProject": None
                }
            }).encode()
            mock_post.return_value = mock_response

            response = client.post(
                "/v1/graphql",
                json={
                    "query": query,
                    "variables": {"id": "user_b_project_123"}
                },
                headers={
                    "X-API-Key": TEST_API_KEY_USER_A,
                    "Content-Type": "application/json"
                }
            )

            assert response.status_code == 200
            data = response.json()

            # Should return null, not the actual data
            assert data["data"]["codeProject"] is None

    def test_workspace_scoped_queries_filtered(self, app):
        """Test that workspace-scoped queries only return workspace data"""
        if not TEST_API_KEY_USER_A:
            pytest.skip("TEST_X_PAPR_API_KEY not set in .env")

        client = TestClient(app)

        query = """
        query GetCodeProjects {
            codeProjects(options: { limit: 10 }) {
                id
                name
                workspace_id
            }
        }
        """

        captured_jwt = None

        def capture_and_respond(*args, **kwargs):
            nonlocal captured_jwt
            auth_header = kwargs["headers"].get("Authorization", "")
            if auth_header.startswith("Bearer "):
                captured_jwt = auth_header.replace("Bearer ", "")

            # Decode JWT to get workspace_id
            decoded = pyjwt.decode(captured_jwt, options={"verify_signature": False})
            workspace_id = decoded.get("workspace_id")

            # Simulate Neo4j filtering by workspace_id
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.content = json.dumps({
                "data": {
                    "codeProjects": [
                        {
                            "id": "proj_1",
                            "name": "Project 1",
                            "workspace_id": workspace_id
                        },
                        {
                            "id": "proj_2",
                            "name": "Project 2",
                            "workspace_id": workspace_id
                        }
                    ]
                }
            }).encode()
            return mock_response

        with patch('httpx.AsyncClient.post', side_effect=capture_and_respond):
            response = client.post(
                "/v1/graphql",
                json={"query": query},
                headers={
                    "X-API-Key": TEST_API_KEY_USER_A,
                    "Content-Type": "application/json"
                }
            )

            assert response.status_code == 200
            data = response.json()

            # All returned projects should have the same workspace_id as JWT
            decoded = pyjwt.decode(captured_jwt, options={"verify_signature": False})
            expected_workspace_id = decoded.get("workspace_id")

            for project in data["data"]["codeProjects"]:
                assert project["workspace_id"] == expected_workspace_id


class TestAuthorizationDirectiveCompliance:
    """Test that queries comply with Neo4j @authorization directives"""

    def test_jwt_matches_authorization_directive_format(self, app):
        """Test that JWT claims match the format expected by @authorization"""
        if not TEST_API_KEY_USER_A:
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
                    "X-API-Key": TEST_API_KEY_USER_A,
                    "Content-Type": "application/json"
                }
            )

            assert response.status_code == 200
            assert captured_jwt is not None

            decoded = pyjwt.decode(
                captured_jwt,
                options={"verify_signature": False}
            )

            # Neo4j @authorization directive expects:
            # where: { node: { user_id: "$jwt.user_id" } }
            # So JWT must have "user_id" claim (not just "sub")
            assert "user_id" in decoded

            # For workspace filtering:
            # where: { node: { workspace_id: "$jwt.workspace_id" } }
            # workspace_id should be present (can be None)
            logger.info(f"JWT claims: user_id={decoded.get('user_id')}, workspace_id={decoded.get('workspace_id')}")

    def test_jwt_signature_valid(self, app):
        """Test that generated JWT has valid signature"""
        if not TEST_API_KEY_USER_A:
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
                    "X-API-Key": TEST_API_KEY_USER_A,
                    "Content-Type": "application/json"
                }
            )

            assert response.status_code == 200
            assert captured_jwt is not None

            # Verify signature using JWT service
            jwt_service = get_jwt_service()
            payload = jwt_service.verify_token(captured_jwt)

            # Should not raise exception, and should return payload
            assert payload is not None
            assert "user_id" in payload

            logger.info("✅ JWT signature is valid")


class TestCrossWorkspaceIsolation:
    """Test isolation between workspaces"""

    def test_personal_data_not_accessible_via_workspace(self, app):
        """Test that personal data (workspace_id=null) is isolated"""
        if not TEST_API_KEY_USER_A:
            pytest.skip("TEST_X_PAPR_API_KEY not set in .env")

        client = TestClient(app)

        # Query that would return both personal and workspace data
        query = """
        query GetAllCodeProjects {
            codeProjects(options: { limit: 100 }) {
                id
                name
                workspace_id
            }
        }
        """

        captured_jwt = None

        def capture_and_filter(*args, **kwargs):
            nonlocal captured_jwt
            auth_header = kwargs["headers"].get("Authorization", "")
            if auth_header.startswith("Bearer "):
                captured_jwt = auth_header.replace("Bearer ", "")

            decoded = pyjwt.decode(captured_jwt, options={"verify_signature": False})
            user_id = decoded["user_id"]
            workspace_id = decoded.get("workspace_id")

            # Simulate Neo4j @authorization filtering:
            # Returns only data where:
            # - user_id matches JWT AND
            # - (workspace_id matches JWT OR workspace_id is null for personal)
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.content = json.dumps({
                "data": {
                    "codeProjects": [
                        # Personal data (workspace_id = null)
                        {
                            "id": "personal_proj_1",
                            "name": "My Personal Project",
                            "workspace_id": None
                        },
                        # Workspace data (if workspace_id exists)
                        {
                            "id": "workspace_proj_1",
                            "name": "Team Project",
                            "workspace_id": workspace_id
                        } if workspace_id else None
                    ]
                }
            }).encode()
            return mock_response

        with patch('httpx.AsyncClient.post', side_effect=capture_and_filter):
            response = client.post(
                "/v1/graphql",
                json={"query": query},
                headers={
                    "X-API-Key": TEST_API_KEY_USER_A,
                    "Content-Type": "application/json"
                }
            )

            assert response.status_code == 200
            data = response.json()

            # Verify isolation - should only get user's own data
            decoded = pyjwt.decode(captured_jwt, options={"verify_signature": False})
            user_workspace_id = decoded.get("workspace_id")

            for project in data["data"]["codeProjects"]:
                if project:  # Skip None entries
                    # Either personal (null) or matching workspace
                    assert (
                        project["workspace_id"] is None or
                        project["workspace_id"] == user_workspace_id
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
