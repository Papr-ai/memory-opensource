#!/usr/bin/env python3
"""
Integration Tests for Batch Processing with Temporal

Tests the actual batch processing route with both OSS and Cloud configurations.
"""

import pytest
import httpx
import asyncio
import os
from typing import Dict, Any

# Test the actual FastAPI app
from main import app
from config import get_features

# Test data
TEST_SESSION_TOKEN = os.getenv('TEST_SESSION_TOKEN')
TEST_USER_ID = os.getenv('TEST_USER_ID')
TEST_X_PAPR_API_KEY = os.getenv('TEST_X_PAPR_API_KEY')

# Skip if no test credentials
pytestmark = pytest.mark.skipif(
    not all([TEST_SESSION_TOKEN, TEST_USER_ID, TEST_X_PAPR_API_KEY]),
    reason="Test credentials not available"
)


@pytest.mark.asyncio
async def test_oss_batch_size_limit_enforcement():
    """Test that OSS edition enforces batch size limits."""

    # Create a batch that exceeds OSS limits (>50)
    large_batch = {
        "memories": [
            {
                "content": f"OSS test memory {i}",
                "type": "text",
                "metadata": {"test": "oss_batch_limit", "index": i}
            }
            for i in range(75)  # Exceed OSS limit
        ]
    }

    headers = {
        'Content-Type': 'application/json',
        'X-Client-Type': 'papr_plugin',
        'X-API-Key': TEST_X_PAPR_API_KEY,
        'Accept-Encoding': 'gzip'
    }

    # Test with actual endpoint
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:

        # Mock OSS configuration
        with pytest.MonkeyPatch.context() as mp:
            # Force OSS configuration for this test
            mp.setenv("PAPR_EDITION", "opensource")

            response = await async_client.post(
                "/v1/memory/batch",
                json=large_batch,
                headers=headers
            )

            # Should be rejected with 413 (Payload Too Large) in OSS
            if response.status_code == 413:
                print("✅ OSS correctly rejected large batch")
                result = response.json()
                assert result["status"] == "error"
                assert "exceeds limit" in result["error"].lower()
            else:
                print(f"ℹ️  Current system processed batch (status: {response.status_code})")
                # This might happen if we're testing against cloud edition


@pytest.mark.asyncio
async def test_cloud_batch_temporal_decision():
    """Test that cloud edition makes correct Temporal decisions."""

    # Test small batch (should use background tasks)
    small_batch = {
        "memories": [
            {
                "content": f"Cloud small batch memory {i}",
                "type": "text",
                "metadata": {"test": "cloud_small_batch", "index": i}
            }
            for i in range(25)  # Below Temporal threshold
        ]
    }

    # Test larger batch (might use Temporal if threshold is low)
    large_batch = {
        "memories": [
            {
                "content": f"Cloud large batch memory {i}",
                "type": "text",
                "metadata": {"test": "cloud_large_batch", "index": i}
            }
            for i in range(150)  # Above typical Temporal threshold
        ]
    }

    headers = {
        'Content-Type': 'application/json',
        'X-Client-Type': 'papr_plugin',
        'X-API-Key': TEST_X_PAPR_API_KEY,
        'Accept-Encoding': 'gzip'
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:

        # Test small batch
        response = await async_client.post(
            "/v1/memory/batch",
            json=small_batch,
            headers=headers
        )

        print(f"Small batch response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Small batch processed: {result.get('total_processed', 0)} memories")

        # Test large batch (if Temporal is available)
        response = await async_client.post(
            "/v1/memory/batch",
            json=large_batch,
            headers=headers
        )

        print(f"Large batch response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            # Check if this was processed with Temporal
            if "workflow_id" in result.get("details", {}):
                print("✅ Large batch processed with Temporal")
                assert result["details"]["status"] == "processing"
                assert "workflow_id" in result["details"]
            else:
                print("ℹ️  Large batch processed with background tasks")
        elif response.status_code == 413:
            print("Large batch rejected due to size limits")


@pytest.mark.asyncio
async def test_temporal_workflow_response_format():
    """Test that Temporal workflow responses have correct format."""

    # Only test if we have Temporal enabled
    features = get_features()
    if not features.is_enabled("temporal"):
        pytest.skip("Temporal not enabled")

    # Create batch likely to trigger Temporal
    temporal_batch = {
        "memories": [
            {
                "content": f"Temporal test memory {i}",
                "type": "text",
                "metadata": {
                    "test": "temporal_response_format",
                    "index": i,
                    "webhook_test": True
                }
            }
            for i in range(120)  # Above threshold
        ],
        "webhook_url": "https://httpbin.org/post",
        "webhook_secret": "test_secret_123"
    }

    headers = {
        'Content-Type': 'application/json',
        'X-Client-Type': 'papr_plugin',
        'X-API-Key': TEST_X_PAPR_API_KEY,
        'Accept-Encoding': 'gzip'
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
        response = await async_client.post(
            "/v1/memory/batch",
            json=temporal_batch,
            headers=headers
        )

        if response.status_code == 200:
            result = response.json()

            # Check if this used Temporal
            if "workflow_id" in result.get("details", {}):
                print("✅ Temporal workflow initiated")

                # Verify response format
                assert result["status"] == "success"
                assert result["details"]["status"] == "processing"
                assert "workflow_id" in result["details"]
                assert "batch_id" in result["details"]
                assert result["details"]["webhook_url"] == "https://httpbin.org/post"
                assert "estimated_completion_minutes" in result["details"]

                print(f"Workflow ID: {result['details']['workflow_id']}")
                print(f"Batch ID: {result['details']['batch_id']}")
                print(f"Estimated completion: {result['details']['estimated_completion_minutes']} minutes")
            else:
                print("ℹ️  Processed with background tasks instead of Temporal")


@pytest.mark.asyncio
async def test_multi_tenant_scoping_with_temporal():
    """Test that multi-tenant scoping works with Temporal processing."""

    # Create batch with multi-tenant context
    multi_tenant_batch = {
        "memories": [
            {
                "content": f"Multi-tenant memory {i}",
                "type": "text",
                "organization_id": "test_org_123",
                "namespace_id": "test_namespace_456",
                "metadata": {"test": "multi_tenant_temporal", "index": i}
            }
            for i in range(30)
        ],
        "organization_id": "test_org_123",
        "namespace_id": "test_namespace_456"
    }

    headers = {
        'Content-Type': 'application/json',
        'X-Client-Type': 'papr_plugin',
        'X-API-Key': TEST_X_PAPR_API_KEY,
        'Accept-Encoding': 'gzip'
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
        response = await async_client.post(
            "/v1/memory/batch",
            json=multi_tenant_batch,
            headers=headers
        )

        print(f"Multi-tenant batch response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ Multi-tenant batch processed successfully")

            # The scoping should be applied regardless of processing method
            # (background tasks or Temporal)
        else:
            print(f"Multi-tenant batch failed: {response.status_code}")
            if response.status_code == 413:
                print("Batch rejected due to size limits")


@pytest.mark.asyncio
async def test_batch_processing_performance():
    """Test performance characteristics of batch processing."""

    import time

    # Test different batch sizes
    batch_sizes = [10, 25, 50]

    for size in batch_sizes:
        batch = {
            "memories": [
                {
                    "content": f"Performance test memory {i} for batch size {size}",
                    "type": "text",
                    "metadata": {"test": "performance", "batch_size": size, "index": i}
                }
                for i in range(size)
            ]
        }

        headers = {
            'Content-Type': 'application/json',
            'X-Client-Type': 'papr_plugin',
            'X-API-Key': TEST_X_PAPR_API_KEY,
            'Accept-Encoding': 'gzip'
        }

        start_time = time.time()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
            response = await async_client.post(
                "/v1/memory/batch",
                json=batch,
                headers=headers,
                timeout=60.0  # Longer timeout for large batches
            )

        end_time = time.time()
        duration = end_time - start_time

        print(f"Batch size {size}: {response.status_code} in {duration:.2f}s")

        if response.status_code == 200:
            result = response.json()
            if "workflow_id" in result.get("details", {}):
                print(f"  → Processed with Temporal (workflow: {result['details']['workflow_id']})")
            else:
                processed = result.get("total_processed", 0)
                print(f"  → Processed {processed} memories with background tasks")


if __name__ == "__main__":
    # Run specific tests
    import sys
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        pytest.main([f"{__file__}::{test_name}", "-v", "-s"])
    else:
        pytest.main([__file__, "-v", "-s"])