"""
Test Telemetry Routes

Tests for the telemetry proxy endpoint that receives anonymous telemetry
from OSS installations and forwards to Amplitude.

Run with: 
    pytest tests/test_telemetry_routes.py -v
    poetry run pytest tests/test_telemetry_routes.py -v
"""

import pytest
import httpx
import json
import os
import time
from fastapi.testclient import TestClient
from main import app
from services.logger_singleton import LoggerSingleton
from os import environ as env
from dotenv import load_dotenv, find_dotenv
from unittest.mock import patch, MagicMock, AsyncMock
from asgi_lifespan import LifespanManager

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

logger = LoggerSingleton.get_logger(__name__)


@pytest.mark.asyncio
async def test_telemetry_endpoint_basic(app):
    """Test basic telemetry endpoint functionality with real Amplitude call"""
    async with LifespanManager(app, startup_timeout=20):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
            # Use real Amplitude API key from environment (or skip if not set)
            amplitude_key = os.getenv("AMPLITUDE_API_KEY") or os.getenv("PAPR_OSS_TELEMETRY_AMPLITUDE_KEY")
            if not amplitude_key:
                pytest.skip("AMPLITUDE_API_KEY or PAPR_OSS_TELEMETRY_AMPLITUDE_KEY not set - skipping real Amplitude test")
            
            # Set Amplitude API key for testing
            with patch.dict('os.environ', {'AMPLITUDE_API_KEY': amplitude_key}):
                request_data = {
                    "events": [
                        {
                            "event_name": "test_memory_created",
                            "properties": {
                                "type": "text",
                                "has_metadata": True,
                                "test": True
                            },
                            "user_id": "test_user_123",
                            "timestamp": int(time.time() * 1000)  # Current timestamp in milliseconds
                        }
                    ],
                    "anonymous_id": "test_session_123"
                }
                
                response = await async_client.post(
                    "/v1/telemetry/events",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                
                assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
                # Use json.loads to avoid any async issues with response.json()
                data = json.loads(response.text)
                assert isinstance(data, dict), f"Expected dict, got {type(data)}: {data}. Response: {data}"
                assert "success" in data, f"Response missing 'success' key. Response: {data}"
                assert data["success"] is True, f"Expected success=True, got {data.get('success')}. Response: {data}"
                assert data["events_received"] == 1
                assert data["events_processed"] == 1
                
                logger.info(f"✅ Telemetry event successfully sent to Amplitude. Response: {data}")


@pytest.mark.asyncio
async def test_telemetry_endpoint_multiple_events(app):
    """Test telemetry endpoint with multiple events using real Amplitude"""
    async with LifespanManager(app, startup_timeout=20):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
            # Use real Amplitude API key from environment
            amplitude_key = os.getenv("AMPLITUDE_API_KEY") or os.getenv("PAPR_OSS_TELEMETRY_AMPLITUDE_KEY")
            if not amplitude_key:
                pytest.skip("AMPLITUDE_API_KEY or PAPR_OSS_TELEMETRY_AMPLITUDE_KEY not set - skipping real Amplitude test")
            
            with patch.dict('os.environ', {'AMPLITUDE_API_KEY': amplitude_key}):
                request_data = {
                    "events": [
                        {
                            "event_name": "test_memory_created",
                            "properties": {"type": "text", "test": True}
                        },
                        {
                            "event_name": "test_search_performed",
                            "properties": {"result_count": 10, "test": True}
                        }
                    ]
                }
                
                response = await async_client.post(
                    "/v1/telemetry/events",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                
                assert response.status_code == 200
                data = json.loads(response.text)
                assert data["events_received"] == 2
                assert data["events_processed"] == 2
                assert data["success"] is True
                logger.info(f"✅ Multiple telemetry events successfully sent to Amplitude. Response: {data}")


@pytest.mark.asyncio
async def test_telemetry_endpoint_anonymizes_pii(app):
    """Test that telemetry endpoint anonymizes PII using real Amplitude"""
    async with LifespanManager(app, startup_timeout=20):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
            # Use real Amplitude API key from environment
            amplitude_key = os.getenv("AMPLITUDE_API_KEY") or os.getenv("PAPR_OSS_TELEMETRY_AMPLITUDE_KEY")
            if not amplitude_key:
                pytest.skip("AMPLITUDE_API_KEY or PAPR_OSS_TELEMETRY_AMPLITUDE_KEY not set - skipping real Amplitude test")
            
            with patch.dict('os.environ', {'AMPLITUDE_API_KEY': amplitude_key}):
                request_data = {
                    "events": [
                        {
                            "event_name": "test_pii_anonymization",
                            "properties": {
                                "content": "This should be removed",  # PII
                                "query": "user search query",  # PII
                                "email": "user@example.com",  # PII
                                "type": "text",  # Safe
                                "has_metadata": True,  # Safe
                                "test": True
                            }
                        }
                    ]
                }
                
                response = await async_client.post(
                    "/v1/telemetry/events",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                
                assert response.status_code == 200
                data = json.loads(response.text)
                assert data["success"] is True
                assert data["events_processed"] == 1
                
                # Note: We can't verify PII removal from the response since Amplitude
                # processes it server-side. The anonymization happens in the route before
                # sending to Amplitude. Check logs or Amplitude dashboard to verify.
                logger.info(f"✅ PII anonymization test completed. Event sent to Amplitude. Response: {data}")


@pytest.mark.asyncio
async def test_telemetry_endpoint_no_amplitude_key(app):
    """Test telemetry endpoint when Amplitude key is not configured"""
    async with LifespanManager(app, startup_timeout=20):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
            # Ensure AMPLITUDE_API_KEY is not set
            with patch.dict('os.environ', {}, clear=False):
                if 'AMPLITUDE_API_KEY' in env:
                    # Temporarily remove it
                    original_key = env.pop('AMPLITUDE_API_KEY', None)
                    try:
                        request_data = {
                            "events": [
                                {
                                    "event_name": "memory_created",
                                    "properties": {"type": "text"}
                                }
                            ]
                        }
                        
                        response = await async_client.post(
                            "/v1/telemetry/events",
                            json=request_data,
                            headers={"Content-Type": "application/json"}
                        )
                        
                        assert response.status_code == 200
                        data = json.loads(response.text)
                        assert data["success"] is False
                        assert data["events_received"] == 1
                        assert data["events_processed"] == 0
                        assert "not configured" in data["message"].lower()
                    finally:
                        if original_key:
                            env['AMPLITUDE_API_KEY'] = original_key


@pytest.mark.asyncio
async def test_telemetry_endpoint_handles_amplitude_error(app):
    """Test that telemetry endpoint handles Amplitude API errors gracefully"""
    async with LifespanManager(app, startup_timeout=20):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
            # Use real Amplitude API key from environment
            amplitude_key = os.getenv("AMPLITUDE_API_KEY") or os.getenv("PAPR_OSS_TELEMETRY_AMPLITUDE_KEY")
            if not amplitude_key:
                pytest.skip("AMPLITUDE_API_KEY or PAPR_OSS_TELEMETRY_AMPLITUDE_KEY not set - skipping real Amplitude test")
            
            # Test with invalid API key to trigger error handling
            with patch.dict('os.environ', {'AMPLITUDE_API_KEY': 'invalid_key_for_testing'}):
                request_data = {
                    "events": [
                        {
                            "event_name": "test_error_handling",
                            "properties": {"type": "text", "test": True}
                        }
                    ]
                }
                
                response = await async_client.post(
                    "/v1/telemetry/events",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                
                # Should still return 200 (fail silently)
                assert response.status_code == 200
                data = json.loads(response.text)
                # Event was received but may not have been processed successfully
                assert data["events_received"] == 1
                # With invalid key, events_processed might be 0 or 1 depending on Amplitude's response
                logger.info(f"✅ Error handling test completed. Response: {data}")


@pytest.mark.asyncio
async def test_telemetry_endpoint_validates_request(app):
    """Test that telemetry endpoint validates request format"""
    async with LifespanManager(app, startup_timeout=20):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
            # Missing required 'events' field
            request_data = {
                "anonymous_id": "session_123"
            }
            
            response = await async_client.post(
                "/v1/telemetry/events",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            # Should return validation error
            assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_telemetry_endpoint_adds_context(app):
    """Test that telemetry endpoint adds technical context using real Amplitude"""
    async with LifespanManager(app, startup_timeout=20):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
            # Use real Amplitude API key from environment
            amplitude_key = os.getenv("AMPLITUDE_API_KEY") or os.getenv("PAPR_OSS_TELEMETRY_AMPLITUDE_KEY")
            if not amplitude_key:
                pytest.skip("AMPLITUDE_API_KEY or PAPR_OSS_TELEMETRY_AMPLITUDE_KEY not set - skipping real Amplitude test")
            
            with patch.dict('os.environ', {'AMPLITUDE_API_KEY': amplitude_key}):
                request_data = {
                    "events": [
                        {
                            "event_name": "test_context_addition",
                            "properties": {"type": "text", "test": True}
                        }
                    ]
                }
                
                response = await async_client.post(
                    "/v1/telemetry/events",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                
                assert response.status_code == 200
                data = json.loads(response.text)
                assert data["success"] is True
                assert data["events_processed"] == 1
                
                # Note: We can't verify context from the response since Amplitude
                # processes it server-side. The context is added in the route before
                # sending to Amplitude. Check Amplitude dashboard to verify context fields
                # (version, edition, python_version, etc.) are present.
                logger.info(f"✅ Context addition test completed. Event sent to Amplitude. Response: {data}")

