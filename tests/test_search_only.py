"""
Simple test for memory search route to debug Groq logging
"""
import pytest
import httpx
from contextlib import asynccontextmanager
from asgi_lifespan import LifespanManager

# Test constants
TEST_X_USER_API_KEY = "sk-papr-test-key-qQKa7NLSPm"

@pytest.fixture
async def app():
    """Create test app instance"""
    from app_factory import create_app
    return create_app()

async def test_search_only_agentic_graph(app, caplog):
    """Test just the search route to see Groq logging"""
    async with LifespanManager(app, startup_timeout=120):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            verify=False,
        ) as async_client:
            headers = {
                'Content-Type': 'application/json',
                'X-Client-Type': 'papr_plugin',
                'X-API-Key': TEST_X_USER_API_KEY,
                'Accept-Encoding': 'gzip'
            }

            target_user_id = "qQKa7NLSPm"

            # Simple search request to trigger Groq logging
            search_request = {
                "query": "Introducing Papr: Predictive Memory Layer that helps AI agents remember",
                "user_id": target_user_id,
                "top_k": 10
            }

            print(f"\n🔍 Testing search with query: {search_request['query']}")

            # Call the search endpoint
            search_response = await async_client.post(
                "/search/v1/agentic_graph",
                json=search_request,
                headers=headers,
                timeout=60.0
            )

            print(f"🔍 Search response status: {search_response.status_code}")
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                print(f"🔍 Search results: {len(search_data.get('memory_nodes', []))} memory nodes, {len(search_data.get('graph_nodes', []))} graph nodes")
            else:
                print(f"🔍 Search error: {search_response.text}")

            # The test passes regardless - we just want to see the logs
            assert True
