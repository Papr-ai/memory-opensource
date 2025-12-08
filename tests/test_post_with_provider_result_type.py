"""
Test that fetch_post_with_provider_result_async returns properly typed responses.
"""
import pytest
from unittest.mock import AsyncMock, patch
from models.parse_server import PostWithProviderResult, PostParseServer


@pytest.mark.asyncio
async def test_fetch_post_returns_typed_response():
    """Test that fetch_post_with_provider_result_async returns PostWithProviderResult."""
    from services.memory_management import fetch_post_with_provider_result_async
    
    # Mock the Parse Server HTTP response
    mock_post_json = {
        "objectId": "test123",
        "organizationId": "org123",
        "namespaceId": "ns123",
        "uploadId": "upload123",
        "post_title": "Test Document",
        "content": {
            "provider": "reducto",
            "provider_result": {"result": {"chunks": []}}
        },
        "file": {
            "__type": "File",
            "name": "doc.pdf",
            "url": "https://example.com/doc.pdf"
        },
        "metadata": {
            "file_url": "https://example.com/doc.pdf",
            "file_name": "doc.pdf"
        },
        "processingMetadata": {
            "total_pages": 10,
            "confidence": 0.95
        },
        "workspace": {
            "__type": "Pointer",
            "className": "WorkSpace",
            "objectId": "workspace123"
        }
    }
    
    # Mock httpx.AsyncClient
    with patch('services.memory_management.httpx.AsyncClient') as mock_client:
        from unittest.mock import Mock
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value=mock_post_json)  # json() is sync in httpx
        
        mock_get = AsyncMock(return_value=mock_response)
        
        mock_client.return_value.__aenter__.return_value.get = mock_get
        
        # Call the function
        result = await fetch_post_with_provider_result_async("test123")
        
        # Assertions
        assert result is not None, "Result should not be None"
        assert isinstance(result, PostWithProviderResult), f"Result should be PostWithProviderResult, got {type(result)}"
        
        # Check that the Post is properly typed
        assert isinstance(result.post, PostParseServer), "post field should be PostParseServer"
        assert result.post.objectId == "test123"
        
        # Check extracted convenience fields
        assert result.organization_id == "org123"
        assert result.namespace_id == "ns123"
        assert result.upload_id == "upload123"
        assert result.post_title == "Test Document"
        assert result.provider_name == "reducto"
        assert result.workspace_id == "workspace123"
        assert result.file_name == "doc.pdf"
        assert result.file_url == "https://example.com/doc.pdf"
        
        # Check metadata dicts
        assert result.processing_metadata["total_pages"] == 10
        assert result.processing_metadata["confidence"] == 0.95
        assert result.file_metadata["file_name"] == "doc.pdf"
        
        # Check provider_specific dict
        assert isinstance(result.provider_specific, dict)
        assert "result" in result.provider_specific
        
        print("✅ TEST PASSED: fetch_post_with_provider_result_async returns properly typed PostWithProviderResult")


@pytest.mark.asyncio
async def test_fetch_post_with_file_download():
    """Test that fetch_post downloads and decompresses provider result files."""
    from services.memory_management import fetch_post_with_provider_result_async
    import json
    
    provider_result = {"result": {"chunks": [{"content": "test"}]}}
    
    mock_post_json = {
        "objectId": "test456",
        "organizationId": "org456",
        "namespaceId": "ns456",
        "post_title": "Test Doc with File",
        "content": {
            "provider": "reducto",
            "provider_result_file": {
                "__type": "File",
                "name": "result.json",
                "url": "https://example.com/result.json"
            }
        },
        "metadata": {},
        "processingMetadata": {}
    }
    
    # Mock httpx.AsyncClient for both Post fetch and file download
    with patch('services.memory_management.httpx.AsyncClient') as mock_client:
        from unittest.mock import Mock
        
        # First call: fetch Post
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json = Mock(return_value=mock_post_json)  # json() is sync in httpx
        
        # Second call: download file
        mock_file_response = AsyncMock()
        mock_file_response.status_code = 200
        mock_file_response.content = json.dumps(provider_result).encode('utf-8')
        
        # Setup the context manager mock
        async def get_side_effect(url, **kwargs):
            if "classes/Post" in url:
                return mock_post_response
            else:
                return mock_file_response
        
        mock_async_context = AsyncMock()
        mock_async_context.get = AsyncMock(side_effect=get_side_effect)
        
        mock_client.return_value.__aenter__.return_value = mock_async_context
        
        # Call the function
        result = await fetch_post_with_provider_result_async("test456")
        
        # Assertions
        assert result is not None
        assert isinstance(result, PostWithProviderResult)
        assert result.provider_specific == provider_result
        assert result.provider_name == "reducto"
        
        print("✅ TEST PASSED: File download and parsing works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

