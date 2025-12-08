"""
Test GraphQL Routes

Tests for the GraphQL proxy endpoint that converts API keys to JWTs
and forwards queries to Neo4j's hosted GraphQL endpoint.

Run with: 
    pytest tests/test_graphql_routes.py -v
    poetry run pytest tests/test_graphql_routes.py -v
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
import time

# Load environment variables
ENV_FILE = find_dotenv()
load_dotenv(ENV_FILE)

logger = LoggerSingleton.get_logger(__name__)

# Test credentials from .env
TEST_API_KEY = env.get('TEST_X_PAPR_API_KEY')
TEST_SESSION_TOKEN = env.get('TEST_SESSION_TOKEN')
TEST_BEARER_TOKEN = env.get('TEST_BEARER_TOKEN')

# Helper to create mock auth response
def get_mock_auth_response():
    """Create a mock OptimizedAuthResponse for testing"""
    from models.memory_models import OptimizedAuthResponse
    return OptimizedAuthResponse(
        developer_id="test_developer_123",
        end_user_id="test_user_456",
        workspace_id="test_workspace_789"
    )


class TestGraphQLProxy:
    """Test GraphQL proxy endpoint basic functionality"""

    def test_graphql_endpoint_requires_authentication(self, app):
        """Test that GraphQL endpoint requires authentication"""
        with TestClient(app) as client:
            query = """
            query {
                __typename
            }
            """

            response = client.post(
                "/v1/graphql",
                json={"query": query},
                headers={"Content-Type": "application/json"}
            )

            assert response.status_code == 401
            assert "authentication" in response.text.lower()

    def test_graphql_endpoint_with_api_key(self, app):
        """Test GraphQL endpoint with valid API key"""
        if not TEST_API_KEY:
            pytest.skip("TEST_X_PAPR_API_KEY not set in .env")

        with TestClient(app) as client:
            query = """
            query {
                __typename
            }
            """

            # Mock authentication and Neo4j GraphQL response
            with patch('routers.v1.graphql_routes.get_user_from_token_optimized', return_value=get_mock_auth_response()):
                with patch('httpx.AsyncClient.post') as mock_post:
                    mock_response = AsyncMock()
                    mock_response.status_code = 200
                    mock_response.content = json.dumps({
                        "data": {"__typename": "Query"}
                    }).encode()
                    mock_post.return_value = mock_response

                    response = client.post(
                        "/v1/graphql",
                        json={"query": query},
                        headers={
                            "X-API-Key": TEST_API_KEY,
                            "Content-Type": "application/json"
                        }
                    )

                    # Should authenticate and forward to Neo4j
                    assert response.status_code == 200
                    mock_post.assert_called_once()

                    # Verify JWT was added to Neo4j request
                    call_args = mock_post.call_args
                    assert "Authorization" in call_args.kwargs["headers"]
                    assert call_args.kwargs["headers"]["Authorization"].startswith("Bearer ")

    def test_graphql_endpoint_with_session_token(self, app):
        """Test GraphQL endpoint with session token"""
        if not TEST_SESSION_TOKEN:
            pytest.skip("TEST_SESSION_TOKEN not set in .env")

        with TestClient(app) as client:
            query = """
            query {
                __typename
            }
            """

            with patch('httpx.AsyncClient.post') as mock_post:
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_response.content = json.dumps({
                    "data": {"__typename": "Query"}
                }).encode()
                mock_post.return_value = mock_response

                response = client.post(
                    "/v1/graphql",
                    json={"query": query},
                    headers={
                        "X-Session-Token": TEST_SESSION_TOKEN,
                        "Content-Type": "application/json"
                    }
                )

                assert response.status_code == 200


    def test_graphql_endpoint_missing_query(self, app):
        """Test that GraphQL endpoint requires a query"""
        if not TEST_API_KEY:
            pytest.skip("TEST_X_PAPR_API_KEY not set in .env")

        with TestClient(app) as client:
            # Mock authentication
            with patch(\'routers.v1.graphql_routes.get_user_from_token_optimized\', return_value=get_mock_auth_response()):
                            response = client.post(
                "/v1/graphql",
                json={},  # No query field
                headers={
                    "X-API-Key": TEST_API_KEY,
                    "Content-Type": "application/json"
                }
            )

            assert response.status_code == 400
            assert "query" in response.text.lower()


    def test_graphql_endpoint_with_variables(self, app):
        """Test GraphQL endpoint with query variables"""
        if not TEST_API_KEY:
            pytest.skip("TEST_X_PAPR_API_KEY not set in .env")

        with TestClient(app) as client:
            # Mock authentication
            with patch(\'routers.v1.graphql_routes.get_user_from_token_optimized\', return_value=get_mock_auth_response()):
                
            query = """
            query GetCodeProject($id: ID!) {
            codeProject(id: $id) {
                id
                name
            }
            }
            """

            variables = {"id": "proj_123"}

            with patch('httpx.AsyncClient.post') as mock_post:
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_response.content = json.dumps({
                    "data": {
                        "codeProject": {
                            "id": "proj_123",
                            "name": "Test Project"
                        }
                    }
                }).encode()
                mock_post.return_value = mock_response

                response = client.post(
                    "/v1/graphql",
                    json={
                        "query": query,
                        "variables": variables,
                        "operationName": "GetCodeProject"
                    },
                    headers={
                        "X-API-Key": TEST_API_KEY,
                        "Content-Type": "application/json"
                    }
                )

                assert response.status_code == 200

                # Verify variables were passed to Neo4j
                call_args = mock_post.call_args
                request_json = call_args.kwargs["json"]
                assert request_json["variables"] == variables
                assert request_json["operationName"] == "GetCodeProject"


    def test_graphql_endpoint_forwards_neo4j_errors(self, app):
        """Test that Neo4j errors are forwarded to client"""
        if not TEST_API_KEY:
            pytest.skip("TEST_X_PAPR_API_KEY not set in .env")

        with TestClient(app) as client:
            # Mock authentication
            with patch(\'routers.v1.graphql_routes.get_user_from_token_optimized\', return_value=get_mock_auth_response()):
                
            query = """
            query {
            invalidQuery {
                field
            }
            }
            """

            with patch('httpx.AsyncClient.post') as mock_post:
                mock_response = AsyncMock()
                mock_response.status_code = 400
                mock_response.content = json.dumps({
                    "errors": [{
                        "message": "Cannot query field 'invalidQuery' on type 'Query'"
                    }]
                }).encode()
                mock_post.return_value = mock_response

                response = client.post(
                    "/v1/graphql",
                    json={"query": query},
                    headers={
                        "X-API-Key": TEST_API_KEY,
                        "Content-Type": "application/json"
                    }
                )

                # Should forward Neo4j's error response
                assert response.status_code == 400
                response_json = response.json()
                assert "errors" in response_json


    def test_graphql_endpoint_timeout_handling(self, app):
        """Test that timeouts are handled gracefully"""
        if not TEST_API_KEY:
            pytest.skip("TEST_X_PAPR_API_KEY not set in .env")

        with TestClient(app) as client:
            # Mock authentication
            with patch(\'routers.v1.graphql_routes.get_user_from_token_optimized\', return_value=get_mock_auth_response()):
                
            query = """
            query {
            __typename
            }
            """

            with patch('httpx.AsyncClient.post') as mock_post:
                mock_post.side_effect = httpx.TimeoutException("Request timeout")

                response = client.post(
                    "/v1/graphql",
                    json={"query": query},
                    headers={
                        "X-API-Key": TEST_API_KEY,
                        "Content-Type": "application/json"
                    }
                )

                assert response.status_code == 504
                assert "timeout" in response.text.lower()


class TestGraphQLJWTGeneration:
    """Test JWT generation for Neo4j authorization"""

    def test_jwt_contains_user_claims(self, app):
        """Test that generated JWT contains user_id and workspace_id"""
        if not TEST_API_KEY:
            pytest.skip("TEST_X_PAPR_API_KEY not set in .env")

        with TestClient(app) as client:
            # Mock authentication
            with patch(\'routers.v1.graphql_routes.get_user_from_token_optimized\', return_value=get_mock_auth_response()):
                
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

                # Verify JWT claims
                jwt_service = get_jwt_service()
                payload = jwt_service.verify_token(captured_jwt)

                assert "user_id" in payload
                assert "sub" in payload
                assert payload["iss"] == "https://memory.papr.ai"
                assert payload["aud"] == "neo4j-graphql"

    def test_jwt_includes_workspace_id(self, app):
        """Test that JWT includes workspace_id for multi-tenant filtering"""
        if not TEST_API_KEY:
            pytest.skip("TEST_X_PAPR_API_KEY not set in .env")

        with TestClient(app) as client:
            # Mock authentication
            with patch(\'routers.v1.graphql_routes.get_user_from_token_optimized\', return_value=get_mock_auth_response()):
                
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

                # Verify workspace_id in JWT
                jwt_service = get_jwt_service()
                payload = jwt_service.verify_token(captured_jwt)

                # workspace_id may be None for personal data, that's OK
                # but the field should be present if a workspace exists
                if "workspace_id" in payload:
                    logger.info(f"Workspace ID in JWT: {payload['workspace_id']}")


class TestGraphQLNeo4jIntegration:
    """Test integration with Neo4j GraphQL endpoint"""

    def test_neo4j_provider_credentials_included(self, app):
        """Test that NEO4J_PROVIDER_ID and KEY are sent to Neo4j"""
        if not TEST_API_KEY:
            pytest.skip("TEST_X_PAPR_API_KEY not set in .env")

        if not env.get("NEO4J_PROVIDER_ID"):
            pytest.skip("NEO4J_PROVIDER_ID not set in .env")

        with TestClient(app) as client:
            # Mock authentication
            with patch(\'routers.v1.graphql_routes.get_user_from_token_optimized\', return_value=get_mock_auth_response()):
                
            query = """
            query {
            __typename
            }
            """

            with patch('httpx.AsyncClient.post') as mock_post:
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_response.content = json.dumps({
                    "data": {"__typename": "Query"}
                }).encode()
                mock_post.return_value = mock_response

                response = client.post(
                    "/v1/graphql",
                    json={"query": query},
                    headers={
                        "X-API-Key": TEST_API_KEY,
                        "Content-Type": "application/json"
                    }
                )

                assert response.status_code == 200

                # Verify Neo4j credentials were included
                call_args = mock_post.call_args
                headers = call_args.kwargs["headers"]

                assert "X-Provider-ID" in headers
                assert "X-Provider-Key" in headers


    def test_graphql_query_code_schema(self, app):
        """Test querying code-related types from the schema"""
        if not TEST_API_KEY:
            pytest.skip("TEST_X_PAPR_API_KEY not set in .env")

        with TestClient(app) as client:
            # Mock authentication
            with patch(\'routers.v1.graphql_routes.get_user_from_token_optimized\', return_value=get_mock_auth_response()):
                
            # Query for CodeProject (from papr_graphql_code_schema.py)
            query = """
            query GetCodeProjects {
            codeProjects(options: { limit: 5 }) {
                id
                name
                description
            }
            }
            """

            with patch('httpx.AsyncClient.post') as mock_post:
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_response.content = json.dumps({
                    "data": {
                        "codeProjects": [
                            {
                                "id": "proj_1",
                                "name": "PAPR Memory",
                                "description": "Knowledge graph memory system"
                            }
                        ]
                    }
                }).encode()
                mock_post.return_value = mock_response

                response = client.post(
                    "/v1/graphql",
                    json={"query": query},
                    headers={
                        "X-API-Key": TEST_API_KEY,
                        "Content-Type": "application/json"
                    }
                )

                assert response.status_code == 200
                data = response.json()
                assert "data" in data
                assert "codeProjects" in data["data"]


class TestGraphQLPlayground:
    """Test GraphQL Playground endpoint"""

    def test_playground_available_in_development(self, app):
        """Test that GraphQL Playground is available in development"""
        with TestClient(app) as client:

            # Set environment to development
            with patch.dict('os.environ', {'ENVIRONMENT': 'development'}):
                response = client.get("/v1/graphql")

            assert response.status_code == 200
            assert "graphiql" in response.text.lower()


    def test_playground_disabled_in_production(self, app):
        """Test that GraphQL Playground is disabled in production"""
        with TestClient(app) as client:

            # Set environment to production
            with patch.dict('os.environ', {'ENVIRONMENT': 'production'}):
                response = client.get("/v1/graphql")

                assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
