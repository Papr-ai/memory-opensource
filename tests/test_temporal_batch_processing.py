#!/usr/bin/env python3
"""
Tests for Temporal Batch Processing

Tests both OSS (background tasks) and Cloud (Temporal) batch processing modes.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Test imports
from services.batch_processor import (
    validate_batch_size,
    should_use_temporal,
    process_batch_with_temporal
)
from models.memory_models import AddMemoryRequest, BatchMemoryRequest
from models.parse_server import BatchMemoryResponse


class TestBatchSizeValidation:
    """Test batch size validation against edition limits."""

    @pytest.mark.asyncio
    async def test_oss_batch_size_validation(self):
        """Test OSS batch size limits."""

        # Mock OSS configuration
        with patch('config.get_features') as mock_features:
            mock_features.return_value.config = {
                "batch_processing": {"max_batch_size": 50}
            }

            # Valid batch size
            is_valid, error_msg, max_allowed = await validate_batch_size(25)
            assert is_valid == True
            assert error_msg == ""
            assert max_allowed == 50

            # Invalid batch size
            is_valid, error_msg, max_allowed = await validate_batch_size(100)
            assert is_valid == False
            assert "exceeds" in error_msg.lower()
            assert max_allowed == 50

    @pytest.mark.asyncio
    async def test_cloud_batch_size_validation(self):
        """Test Cloud batch size limits."""

        # Mock Cloud configuration
        with patch('config.get_features') as mock_features:
            mock_features.return_value.config = {
                "batch_processing": {"max_batch_size": 10000}
            }

            # Valid large batch size
            is_valid, error_msg, max_allowed = await validate_batch_size(5000)
            assert is_valid == True
            assert error_msg == ""
            assert max_allowed == 10000

            # Still enforce some limit
            is_valid, error_msg, max_allowed = await validate_batch_size(20000)
            assert is_valid == False
            assert max_allowed == 10000


class TestTemporalDecision:
    """Test when to use Temporal vs background tasks."""

    @pytest.mark.asyncio
    async def test_oss_no_temporal(self):
        """Test OSS edition never uses Temporal."""

        with patch('config.get_features') as mock_features:
            mock_features.return_value.is_enabled.return_value = False

            # Even large batches don't use Temporal in OSS
            use_temporal = await should_use_temporal(1000)
            assert use_temporal == False

    @pytest.mark.asyncio
    async def test_cloud_temporal_threshold(self):
        """Test Cloud edition Temporal threshold."""

        with patch('config.get_features') as mock_features:
            mock_features.return_value.is_enabled.return_value = True
            mock_features.return_value.config = {
                "temporal": {"temporal_threshold": 100}
            }

            # Small batch - no Temporal
            use_temporal = await should_use_temporal(50)
            assert use_temporal == False

            # Large batch - use Temporal
            use_temporal = await should_use_temporal(150)
            assert use_temporal == True

    @pytest.mark.asyncio
    async def test_cloud_temporal_disabled(self):
        """Test Cloud edition with Temporal disabled."""

        with patch('config.get_features') as mock_features:
            mock_features.return_value.is_enabled.return_value = False

            # Even in cloud, if Temporal disabled, don't use it
            use_temporal = await should_use_temporal(1000)
            assert use_temporal == False


class TestTemporalWorkflowIntegration:
    """Test Temporal workflow integration."""

    @pytest.mark.asyncio
    async def test_temporal_workflow_start(self):
        """Test starting Temporal workflow for batch processing."""

        memories_data = [
            {"content": "Test memory 1", "type": "text"},
            {"content": "Test memory 2", "type": "text"}
        ]

        mock_client = AsyncMock()
        mock_handle = AsyncMock()
        mock_client.start_workflow.return_value = mock_handle

        with patch('cloud_plugins.temporal.client.get_temporal_client', return_value=mock_client):
            result = await process_batch_with_temporal(
                memories=memories_data,
                developer_id="dev_123",
                workspace_id="workspace_456",
                webhook_url="https://example.com/webhook",
                webhook_secret="secret_key"
            )

            # Verify workflow was started
            mock_client.start_workflow.assert_called_once()

            # Verify response format
            assert result.status == "success"
            assert "workflow_id" in result.details
            assert result.details["status"] == "processing"
            assert result.details["webhook_url"] == "https://example.com/webhook"

    @pytest.mark.asyncio
    async def test_temporal_unavailable_fallback(self):
        """Test fallback when Temporal is unavailable."""

        memories_data = [
            {"content": "Test memory 1", "type": "text"}
        ]

        # Mock Temporal import error
        with patch('cloud_plugins.temporal.client.get_temporal_client', side_effect=ImportError("Temporal not available")):
            result = await process_batch_with_temporal(
                memories=memories_data,
                developer_id="dev_123",
                workspace_id="workspace_456"
            )

            # Verify error response
            assert result.status == "error"
            assert result.code == 503
            assert "unavailable" in result.error.lower()


class TestBatchProcessingEndToEnd:
    """End-to-end tests for batch processing routes."""

    def create_test_batch_request(self, num_memories: int) -> BatchMemoryRequest:
        """Helper to create test batch requests."""
        memories = []
        for i in range(num_memories):
            memories.append(AddMemoryRequest(
                content=f"Test memory {i+1}",
                type="text"
            ))

        return BatchMemoryRequest(
            memories=memories,
            webhook_url="https://example.com/webhook"
        )

    @pytest.mark.asyncio
    async def test_oss_small_batch_background_tasks(self):
        """Test OSS edition processes small batches with background tasks."""

        batch_request = self.create_test_batch_request(25)

        with patch('config.get_features') as mock_features:
            # Mock OSS configuration
            mock_features.return_value.is_enabled.return_value = False
            mock_features.return_value.config = {
                "batch_processing": {"max_batch_size": 50}
            }

            with patch('services.batch_processor.should_use_temporal', return_value=False):
                with patch('routes.memory_routes.common_add_memory_batch_handler') as mock_handler:
                    mock_handler.return_value = AsyncMock()
                    mock_handler.return_value.status = "success"

                    # This would be called from the route
                    use_temporal = await should_use_temporal(len(batch_request.memories))
                    assert use_temporal == False

    @pytest.mark.asyncio
    async def test_oss_large_batch_rejected(self):
        """Test OSS edition rejects large batches."""

        batch_request = self.create_test_batch_request(100)

        with patch('config.get_features') as mock_features:
            # Mock OSS configuration
            mock_features.return_value.config = {
                "batch_processing": {"max_batch_size": 50},
                "messaging": {"batch_limit_exceeded": "Batch size exceeds open source limits"}
            }

            # Validate batch size
            is_valid, error_msg, max_allowed = await validate_batch_size(len(batch_request.memories))
            assert is_valid == False
            assert "exceeds" in error_msg

    @pytest.mark.asyncio
    async def test_cloud_small_batch_background_tasks(self):
        """Test Cloud edition processes small batches with background tasks."""

        batch_request = self.create_test_batch_request(50)

        with patch('config.get_features') as mock_features:
            # Mock Cloud configuration
            mock_features.return_value.is_enabled.return_value = True
            mock_features.return_value.config = {
                "batch_processing": {"max_batch_size": 10000},
                "temporal": {"temporal_threshold": 100}
            }

            # Should not use Temporal for small batches
            use_temporal = await should_use_temporal(len(batch_request.memories))
            assert use_temporal == False

    @pytest.mark.asyncio
    async def test_cloud_large_batch_temporal(self):
        """Test Cloud edition processes large batches with Temporal."""

        batch_request = self.create_test_batch_request(200)

        with patch('config.get_features') as mock_features:
            # Mock Cloud configuration
            mock_features.return_value.is_enabled.return_value = True
            mock_features.return_value.config = {
                "batch_processing": {"max_batch_size": 10000},
                "temporal": {"temporal_threshold": 100}
            }

            # Should use Temporal for large batches
            use_temporal = await should_use_temporal(len(batch_request.memories))
            assert use_temporal == True

    @pytest.mark.asyncio
    async def test_webhook_data_included(self):
        """Test webhook data is properly included in Temporal processing."""

        memories_data = [{"content": "Test memory", "type": "text"}]
        webhook_url = "https://example.com/webhook"
        webhook_secret = "my_secret"

        mock_client = AsyncMock()
        mock_handle = AsyncMock()
        mock_client.start_workflow.return_value = mock_handle

        with patch('cloud_plugins.temporal.client.get_temporal_client', return_value=mock_client):
            result = await process_batch_with_temporal(
                memories=memories_data,
                developer_id="dev_123",
                workspace_id="workspace_456",
                webhook_url=webhook_url,
                webhook_secret=webhook_secret
            )

            # Verify webhook data is passed to workflow
            call_args = mock_client.start_workflow.call_args[0]
            workflow_data = call_args[0]

            assert workflow_data["webhook_url"] == webhook_url
            assert workflow_data["webhook_secret"] == webhook_secret


class TestTemporalActivities:
    """Test Temporal activities."""

    @pytest.mark.asyncio
    async def test_process_memory_batch_activity(self):
        """Test the process_memory_batch activity."""

        from cloud_plugins.temporal.activities.memory_activities import process_memory_batch

        batch_data = {
            "batch_id": "test_batch_123",
            "memories": [
                {"content": "Test memory 1", "type": "text"},
                {"content": "Test memory 2", "type": "text"}
            ],
            "developer_id": "dev_123",
            "workspace_id": "workspace_456"
        }

        # Mock dependencies
        with patch('memory.memory_graph.MemoryGraph') as mock_memory_graph:
            with patch('routes.memory_routes.common_add_memory_handler') as mock_handler:
                mock_graph_instance = AsyncMock()
                mock_memory_graph.return_value = mock_graph_instance

                mock_result = AsyncMock()
                mock_result.status = "success"
                mock_handler.return_value = mock_result

                # Mock Temporal activity context
                with patch('temporalio.activity.heartbeat'):
                    result = await process_memory_batch(batch_data)

                assert result["status"] in ["completed", "completed_with_errors"]
                assert result["total_processed"] == 2
                assert result["successful"] >= 0
                assert result["failed"] >= 0

    @pytest.mark.asyncio
    async def test_send_webhook_notification_activity(self):
        """Test the send_webhook_notification activity."""

        from cloud_plugins.temporal.activities.memory_activities import send_webhook_notification

        webhook_data = {
            "batch_id": "test_batch_123",
            "webhook_url": "https://httpbin.org/post",
            "webhook_secret": "test_secret",
            "results": {
                "status": "completed",
                "total_processed": 5,
                "successful": 5,
                "failed": 0
            }
        }

        # Mock activity info
        with patch('temporalio.activity.info') as mock_info:
            mock_info.return_value.current_attempt_scheduled_time.isoformat.return_value = "2023-01-01T00:00:00Z"

            # Mock HTTP client
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

                result = await send_webhook_notification(webhook_data)

                assert result["status"] == "success"
                assert result["response_code"] == 200


class TestBatchAddMemoryQuick:
    """Test the batch_add_memory_quick activity directly."""

    @pytest.mark.asyncio
    async def test_batch_add_memory_quick_activity(self):
        """Test batch_add_memory_quick processes multiple memories in one transaction."""
        
        from cloud_plugins.temporal.activities.memory_activities import batch_add_memory_quick
        from models.memory_models import OptimizedAuthResponse
        
        # Create batch data for 3 memories
        batch_data_list = []
        for i in range(3):
            batch_data = {
                "batch_data": {
                    "batch_id": f"test_batch_{i}",
                    "api_key": "test_api_key",
                    "legacy_route": True,
                    "auth_response": {
                        "developer_id": "test_dev",
                        "end_user_id": "test_user",
                        "workspace_id": "test_workspace",
                        "organization_id": "test_org",
                        "namespace_id": "test_namespace",
                        "is_qwen_route": False
                    },
                    "batch_request": {
                        "user_id": "test_user",
                        "external_user_id": "test_external_user",
                        "organization_id": "test_org",
                        "namespace_id": "test_namespace",
                        "memories": [
                            {
                                "content": f"Test batch memory {i+1}: This is a test of the batch processing system",
                                "type": "text",
                                "metadata": {
                                    "topics": "testing, batch processing",
                                    "test_id": "batch_quick_test"
                                }
                            }
                        ]
                    }
                },
                "index": 0
            }
            batch_data_list.append(batch_data)
        
        # Call the activity directly
        try:
            results = await batch_add_memory_quick(batch_data_list)
            
            # Verify results
            assert len(results) == 3, f"Expected 3 results, got {len(results)}"
            
            for i, result in enumerate(results):
                print(f"Result {i}: {result}")
                assert result is not None, f"Result {i} is None"
                assert "memory_id" in result or result.get("memory_id") is not None, f"Result {i} missing memory_id"
                
            print(f"✅ batch_add_memory_quick successfully processed {len(results)} memories")
            
        except Exception as e:
            pytest.fail(f"batch_add_memory_quick failed: {e}")
    
    @pytest.mark.asyncio
    async def test_batch_add_memory_quick_with_real_data(self):
        """Test batch_add_memory_quick with real data from Temporal workflow."""
        import json
        import os
        from cloud_plugins.temporal.activities.memory_activities import batch_add_memory_quick
        
        # Load real data from fixed JSON file
        data_file = "tests/batch_quick_add_test_data_fixed.json"
        if not os.path.exists(data_file):
            pytest.skip(f"Real test data file not found: {data_file}")
        
        print(f"\n📁 Loading real test data from {data_file}...")
        with open(data_file, 'r') as f:
            batch_request_data = json.load(f)
        
        # Convert to the format expected by batch_add_memory_quick
        # The activity expects a list of {"batch_data": {...}, "index": N} items
        batch_data_list = []
        
        memories = batch_request_data["batch_request"]["memories"]
        print(f"   Found {len(memories)} memories in test data")
        
        # Process only first 10 memories for this test (one batch)
        test_batch_size = min(10, len(memories))
        print(f"   Testing with first {test_batch_size} memories")
        
        # Build the auth_response from batch_request data
        auth_response = {
            "developer_id": batch_request_data["batch_request"]["user_id"],
            "end_user_id": batch_request_data["batch_request"]["user_id"],
            "workspace_id": "85DF0bmVaO",  # From your test env
            "organization_id": batch_request_data["batch_request"]["organization_id"],
            "namespace_id": batch_request_data["batch_request"]["namespace_id"],
            "is_qwen_route": False
        }
        
        # Package ALL memories into SEPARATE batch_data items (one memory per item)
        # This mimics how the workflow packages them
        # NOTE: No need to clean memories here - batch_add_memory_quick handles it internally
        test_memories = memories[:test_batch_size]
        for idx in range(test_batch_size):
            batch_data = {
                "batch_data": {
                    "batch_id": batch_request_data["batch_id"],
                    "api_key": "temporal_internal",
                    "legacy_route": True,
                    "auth_response": auth_response,
                    "batch_request": {
                        "user_id": batch_request_data["batch_request"]["user_id"],
                        "external_user_id": batch_request_data["batch_request"]["external_user_id"],
                        "organization_id": batch_request_data["batch_request"]["organization_id"],
                        "namespace_id": batch_request_data["batch_request"]["namespace_id"],
                        "memories": test_memories  # ALL memories in each item
                    }
                },
                "index": idx  # Index into the memories array
            }
            batch_data_list.append(batch_data)
        
        # Call the activity with real data
        print(f"\n🚀 Calling batch_add_memory_quick with {len(batch_data_list)} real memories...")
        try:
            results = await batch_add_memory_quick(batch_data_list)
            
            # Verify results
            assert len(results) == test_batch_size, f"Expected {test_batch_size} results, got {len(results)}"
            
            successful = 0
            for i, result in enumerate(results):
                print(f"\nResult {i}:")
                print(f"   memory_id: {result.get('memory_id')}")
                print(f"   object_id: {result.get('object_id')}")
                print(f"   batch_id: {result.get('batch_id')}")
                print(f"   memory_chunk_ids: {len(result.get('memory_chunk_ids', []))} chunks")
                
                assert result is not None, f"Result {i} is None"
                
                # Check if memory was successfully created
                if result.get("memory_id"):
                    successful += 1
            
            print(f"\n✅ batch_add_memory_quick completed!")
            print(f"   Total processed: {len(results)}")
            print(f"   Successful: {successful}/{len(results)}")
            print(f"   Success rate: {(successful/len(results)*100):.1f}%")
            
            # We expect at least some successes with real data
            assert successful > 0, f"No memories were successfully created!"
            
        except Exception as e:
            pytest.fail(f"batch_add_memory_quick with real data failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])