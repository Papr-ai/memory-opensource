#!/usr/bin/env python3
"""
Unit tests for BatchMemoryRequest storage functions

Tests the new Parse File storage pattern for batch memory processing
that avoids Temporal gRPC payload limits.
"""

import pytest
import json
import gzip
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List

# Test imports
from services.memory_management import (
    create_batch_memory_request_in_parse,
    fetch_batch_memory_request_from_parse,
    update_batch_request_status
)
from models.parse_server import BatchMemoryRequestParse
from models.memory_models import BatchMemoryRequest, AddMemoryRequest
from models.shared_types import MemoryMetadata

class TestBatchMemoryRequestCreation:
    """Test creating BatchMemoryRequestParse in Parse with compression"""

    @pytest.mark.asyncio
    async def test_create_batch_request_basic(self):
        """Test basic batch request creation with compression"""
        memories = [
            {"content": f"Test memory {i}", "type": "text"}
            for i in range(10)
        ]

        batch_request = Mock()
        batch_request.memories = [Mock(**mem) for mem in memories]
        for i, mem in enumerate(batch_request.memories):
            mem.model_dump = Mock(return_value=memories[i])

        # Mock auth response
        auth_response = Mock()
        auth_response.organization_id = "test_org_123"
        auth_response.namespace_id = "test_ns_123"
        auth_response.end_user_id = "test_user_123"
        auth_response.workspace_id = "test_workspace_123"

        # Mock HTTP client
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"objectId": "batch_req_123"}

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            # Create batch request
            request_id = await create_batch_memory_request_in_parse(
                batch_request=batch_request,
                auth_response=auth_response,
                batch_id="test_batch_001",
                api_key="test_api_key"
            )

            # Assertions
            assert request_id == "batch_req_123"
            assert mock_client.return_value.__aenter__.return_value.post.called

    @pytest.mark.asyncio
    async def test_create_batch_request_compression(self):
        """Test that batch data is properly compressed"""

        # Create large batch to test compression
        memories = [
            {
                "content": f"This is a longer test memory content {i} " * 50,
                "type": "text",
                "metadata": {"index": i}
            }
            for i in range(100)
        ]

        batch_request = Mock()
        batch_request.memories = [Mock(**mem) for mem in memories]
        for i, mem in enumerate(batch_request.memories):
            mem.model_dump = Mock(return_value=memories[i])

        auth_response = Mock()
        auth_response.organization_id = "test_org"
        auth_response.end_user_id = "test_user"

        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"objectId": "batch_req_compressed"}

        posted_data = {}

        async def capture_post(*args, **kwargs):
            posted_data.update(kwargs.get('json', {}))
            return mock_response

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = capture_post

            request_id = await create_batch_memory_request_in_parse(
                batch_request=batch_request,
                auth_response=auth_response,
                batch_id="test_batch_compression",
                api_key="test_key"
            )

            # Verify compression metadata
            assert "batchMetadata" in posted_data
            metadata = posted_data["batchMetadata"]
            assert "compression_ratio" in metadata
            assert metadata["compression_ratio"] > 1.0  # Should be compressed
            assert "compressed_size_bytes" in metadata
            assert "total_size_bytes" in metadata

            # Compressed size should be less than original
            assert metadata["compressed_size_bytes"] < metadata["total_size_bytes"]

    @pytest.mark.asyncio
    async def test_create_batch_request_with_webhook(self):
        """Test batch request creation with webhook configuration"""

        batch_request = Mock()
        batch_request.memories = [Mock(model_dump=Mock(return_value={"content": "test"}))]

        auth_response = Mock()
        auth_response.organization_id = "org"
        auth_response.end_user_id = "user"

        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"objectId": "batch_webhook"}

        posted_data = {}

        async def capture_post(*args, **kwargs):
            posted_data.update(kwargs.get('json', {}))
            return mock_response

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = capture_post

            await create_batch_memory_request_in_parse(
                batch_request=batch_request,
                auth_response=auth_response,
                batch_id="test_webhook",
                webhook_url="https://example.com/webhook",
                webhook_secret="secret123",
                api_key="key"
            )

            # Verify webhook fields
            assert posted_data.get("webhookUrl") == "https://example.com/webhook"
            assert posted_data.get("webhookSecret") == "secret123"
            assert posted_data.get("webhookSent") == False


class TestBatchMemoryRequestFetch:
    """Test fetching and decompressing BatchMemoryRequest"""

    @pytest.mark.asyncio
    async def test_fetch_batch_request_success(self):
        """Test successful batch request fetch with decompression"""

        # Create test data
        test_memories = [
            {"content": f"Memory {i}", "type": "text"}
            for i in range(50)
        ]

        batch_data = {
            "memories": test_memories,
            "batch_metadata": {"source": "test"}
        }

        # Compress the data
        json_str = json.dumps(batch_data)
        compressed = gzip.compress(json_str.encode('utf-8'))

        # Mock Parse response
        parse_response_data = {
            "objectId": "batch_123",
            "batchId": "test_batch",
            "status": "pending",
            "totalMemories": 50,
            "batchDataFile": {
                "name": "batch_test_batch.json.gz",
                "url": "https://files.example.com/batch.json.gz",
                "__type": "File"
            },
            "organization": {
                "objectId": "org_123",
                "__type": "Pointer",
                "className": "Organization"
            }
        }

        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = parse_response_data

        mock_file_response = Mock()
        mock_file_response.status_code = 200
        mock_file_response.content = compressed

        with patch('httpx.AsyncClient') as mock_client:
            async def mock_get(url, **kwargs):
                if "BatchMemoryRequest" in url:
                    return mock_get_response
                else:
                    return mock_file_response

            mock_client.return_value.__aenter__.return_value.get = mock_get

            # Fetch batch request
            result = await fetch_batch_memory_request_from_parse(
                request_id="batch_123",
                api_key="test_key"
            )

            # Assertions
            assert result is not None
            assert result.objectId == "batch_123"
            assert hasattr(result, 'memories')
            assert len(result.memories) == 50
            assert result.memories[0]["content"] == "Memory 0"

    @pytest.mark.asyncio
    async def test_fetch_batch_request_not_found(self):
        """Test fetching non-existent batch request"""

        mock_response = Mock()
        mock_response.status_code = 404

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            # Should return None for not found
            result = await fetch_batch_memory_request_from_parse(
                request_id="nonexistent",
                api_key="test_key"
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_fetch_batch_request_decompression_error(self):
        """Test handling of decompression errors"""

        parse_response_data = {
            "objectId": "batch_123",
            "batchId": "test_batch",
            "status": "pending",
            "totalMemories": 0,
            "batchDataFile": {
                "name": "batch_test.json",
                "url": "https://files.example.com/batch.json",
                "__type": "File"
            }
        }

        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = parse_response_data

        # Return invalid compressed data
        mock_file_response = Mock()
        mock_file_response.status_code = 200
        mock_file_response.content = b"not gzipped data"

        with patch('httpx.AsyncClient') as mock_client:
            async def mock_get(url, **kwargs):
                if "BatchMemoryRequest" in url:
                    return mock_get_response
                else:
                    return mock_file_response

            mock_client.return_value.__aenter__.return_value.get = mock_get

            # Should handle decompression error gracefully
            result = await fetch_batch_memory_request_from_parse(
                request_id="batch_123",
                api_key="test_key"
            )

            assert result is None  # Returns None on error


class TestBatchMemoryRequestStatusUpdate:
    """Test updating BatchMemoryRequest status"""

    @pytest.mark.asyncio
    async def test_update_status_to_processing(self):
        """Test updating status to processing"""

        mock_response = Mock()
        mock_response.status_code = 200

        updated_data = {}

        async def capture_put(*args, **kwargs):
            updated_data.update(kwargs.get('json', {}))
            return mock_response

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.put = capture_put

            success = await update_batch_request_status(
                request_id="batch_123",
                status="processing",
                api_key="test_key"
            )

            # Assertions
            assert success == True
            assert updated_data["status"] == "processing"
            assert "startedAt" in updated_data
            assert updated_data["startedAt"]["__type"] == "Date"

    @pytest.mark.asyncio
    async def test_update_status_to_completed(self):
        """Test updating status to completed with counts"""

        mock_response = Mock()
        mock_response.status_code = 200

        updated_data = {}

        async def capture_put(*args, **kwargs):
            updated_data.update(kwargs.get('json', {}))
            return mock_response

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.put = capture_put

            success = await update_batch_request_status(
                request_id="batch_123",
                status="completed",
                processed_count=100,
                success_count=98,
                fail_count=2,
                errors=[
                    {"index": 5, "error": "Test error"},
                    {"index": 10, "error": "Another error"}
                ],
                api_key="test_key"
            )

            # Assertions
            assert success == True
            assert updated_data["status"] == "completed"
            assert updated_data["processedCount"] == 100
            assert updated_data["successCount"] == 98
            assert updated_data["failCount"] == 2
            assert len(updated_data["errors"]) == 2
            assert "completedAt" in updated_data

    @pytest.mark.asyncio
    async def test_update_status_failure(self):
        """Test handling of update failure"""

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.put = AsyncMock(
                return_value=mock_response
            )

            success = await update_batch_request_status(
                request_id="batch_123",
                status="completed",
                api_key="test_key"
            )

            # Should return False on error
            assert success == False


class TestBatchMemoryRequestIntegration:
    """Integration tests for full create-fetch-update cycle"""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test complete lifecycle: create -> fetch -> update"""

        # This would be a real integration test with actual Parse Server
        # For now, we'll skip it unless Parse Server is available
        pytest.skip("Requires real Parse Server - run manually")

    @pytest.mark.asyncio
    async def test_compression_ratio_validation(self):
        """Test that compression achieves expected ratio"""

        # Create test data with repetitive content (compresses well)
        memories = [
            {
                "content": "This is test content " * 100,
                "type": "text",
                "metadata": {"index": i}
            }
            for i in range(100)
        ]

        # Calculate expected compression
        json_str = json.dumps({"memories": memories})
        original_size = len(json_str.encode('utf-8'))
        compressed_size = len(gzip.compress(json_str.encode('utf-8')))
        compression_ratio = original_size / compressed_size

        # Should achieve at least 5x compression on repetitive text
        assert compression_ratio >= 5.0

        print(f"\n✅ Compression test: {original_size} → {compressed_size} bytes")
        print(f"   Ratio: {compression_ratio:.1f}x")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
