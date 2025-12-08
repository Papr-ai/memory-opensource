"""
Test extraction result storage in Parse Server for large documents
"""
import pytest
import json
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from os import environ as env
from models.memory_models import AddMemoryRequest
from models.shared_types import MemoryMetadata
from models.hierarchical_models import TextElement, ContentType

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)


@pytest.fixture
def large_extraction_data():
    """Create a large extraction result that exceeds 500KB threshold"""
    # Create 1000+ text elements (simulate large document)
    structured_elements = []
    memory_requests = []
    
    for i in range(1000):
        # Each element ~1KB
        content = f"Element {i}: " + ("x" * 900)
        
        elem = TextElement(
            element_id=f"large_elem_{i}",
            content=content,
            metadata={
                "chunk_index": i,
                "provider": "reducto",
                "organization_id": "test_org",
                "namespace_id": "test_ns"
            }
        )
        structured_elements.append(elem)
        
        mem_req = AddMemoryRequest(
            content=content,
            metadata=MemoryMetadata(
                organization_id="test_org",
                namespace_id="test_ns"
            )
        )
        memory_requests.append(mem_req)
    
    return structured_elements, memory_requests


@pytest.fixture
def small_extraction_data():
    """Create a small extraction result that stays under 500KB threshold"""
    structured_elements = []
    memory_requests = []
    
    for i in range(10):
        content = f"Small element {i}: test content"
        
        elem = TextElement(
            element_id=f"small_elem_{i}",
            content=content,
            metadata={
                "chunk_index": i,
                "provider": "reducto"
            }
        )
        structured_elements.append(elem)
        
        mem_req = AddMemoryRequest(
            content=content,
            metadata=MemoryMetadata(
                organization_id="test_org",
                namespace_id="test_ns"
            )
        )
        memory_requests.append(mem_req)
    
    return structured_elements, memory_requests


@pytest.mark.asyncio
async def test_small_extraction_inline_payload():
    """Test that small extractions use inline payload (no Parse storage)"""
    from cloud_plugins.temporal.activities.document_activities import extract_structured_content_from_provider
    
    # Create small provider response (under threshold)
    small_provider = {
        "result": {
            "parse": {
                "result": {
                    "chunks": [
                        {
                            "blocks": [
                                {"content": "Small text block", "type": "Text", "confidence": "high"}
                            ]
                        }
                    ]
                }
            }
        }
    }
    
    base_metadata = MemoryMetadata(
        organization_id="test_org",
        namespace_id="test_ns",
        user_id="test_user"
    )
    
    result = await extract_structured_content_from_provider(
        provider_specific=small_provider,
        provider_name="reducto",
        base_metadata=base_metadata,
        organization_id="test_org",
        namespace_id="test_ns"
    )
    
    # Should use inline payload
    assert result["extraction_stored"] is False
    assert "structured_elements" in result
    assert "memory_requests" in result
    assert len(result["structured_elements"]) > 0 or len(result["memory_requests"]) > 0
    
    print(f"\n✅ Small extraction used inline payload (no Parse storage)")
    print(f"   - Structured elements: {len(result['structured_elements'])}")
    print(f"   - Memory requests: {len(result['memory_requests'])}")


@pytest.mark.asyncio
async def test_large_extraction_parse_storage(large_extraction_data):
    """Test that large extractions are stored in Parse Server"""
    from services.memory_management import store_extraction_result_in_post, fetch_extraction_result_from_post
    
    structured_elements, memory_requests = large_extraction_data
    
    # Convert to dicts for storage
    elem_dicts = [e.model_dump() for e in structured_elements]
    mem_dicts = [m.model_dump() for m in memory_requests]
    
    element_summary = {"text": len(structured_elements)}
    
    # Mock Parse Server calls
    mock_post_id = "test_post_12345"
    mock_file_name = f"extraction_{mock_post_id}_abc123.json.gz"
    mock_file_url = f"https://parsefiles.example.com/{mock_file_name}"
    
    with patch('services.memory_management.httpx.AsyncClient') as mock_client_class:
        # Mock file upload response
        mock_file_response = Mock()
        mock_file_response.status_code = 201
        mock_file_response.json.return_value = {
            "name": mock_file_name,
            "url": mock_file_url
        }
        
        # Mock Post update response
        mock_update_response = Mock()
        mock_update_response.status_code = 200
        
        # Setup mock client
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_file_response
        mock_client.put.return_value = mock_update_response
        mock_client_class.return_value = mock_client
        
        # Test storage
        result_id = await store_extraction_result_in_post(
            post_id=mock_post_id,
            structured_elements=elem_dicts,
            memory_requests=mem_dicts,
            element_summary=element_summary,
            decision="complex"
        )
        
        # Verify storage was called
        assert result_id == mock_file_name
        assert mock_client.post.called  # File upload
        assert mock_client.put.called   # Post update
        
        print(f"\n✅ Large extraction stored in Parse Server")
        print(f"   - Elements: {len(structured_elements)}")
        print(f"   - File: {mock_file_name}")
        
        # Verify compression happened
        post_call_args = mock_client.post.call_args
        uploaded_content = post_call_args[1]["content"]
        original_size = len(json.dumps({"structured_elements": elem_dicts, "memory_requests": mem_dicts}))
        compressed_size = len(uploaded_content)
        compression_ratio = compressed_size / original_size
        
        print(f"   - Original size: {original_size:,} bytes")
        print(f"   - Compressed size: {compressed_size:,} bytes")
        print(f"   - Compression ratio: {compression_ratio:.1%}")
        
        assert compressed_size < original_size, "Should be compressed"


@pytest.mark.asyncio
async def test_extraction_result_roundtrip():
    """Test storing and fetching extraction result (full roundtrip)"""
    from services.memory_management import store_extraction_result_in_post, fetch_extraction_result_from_post
    import gzip
    
    # Create test data
    test_elements = [
        {"element_id": "test1", "content": "Test content 1", "content_type": "text"},
        {"element_id": "test2", "content": "Test content 2", "content_type": "text"}
    ]
    test_memories = [
        {"content": "Memory 1", "type": "document"},
        {"content": "Memory 2", "type": "document"}
    ]
    test_summary = {"text": 2}
    
    mock_post_id = "roundtrip_test_post"
    mock_file_url = "https://parsefiles.example.com/extraction_test.json.gz"
    
    # Prepare what should be returned from fetch
    extraction_data = {
        "decision": "complex",
        "structured_elements": test_elements,
        "memory_requests": test_memories,
        "element_summary": test_summary,
        "extracted_at": "2025-10-20T00:00:00Z"
    }
    compressed_data = gzip.compress(json.dumps(extraction_data).encode('utf-8'))
    
    with patch('services.memory_management.httpx.AsyncClient') as mock_client_class:
        # Mock for store operation
        mock_store_client = AsyncMock()
        mock_store_client.__aenter__.return_value = mock_store_client
        
        mock_file_resp = Mock()
        mock_file_resp.status_code = 201
        mock_file_resp.json.return_value = {"name": "extraction_test.json.gz", "url": mock_file_url}
        
        mock_update_resp = Mock()
        mock_update_resp.status_code = 200
        
        mock_store_client.post.return_value = mock_file_resp
        mock_store_client.put.return_value = mock_update_resp
        
        # Mock for fetch operation
        mock_fetch_client = AsyncMock()
        mock_fetch_client.__aenter__.return_value = mock_fetch_client
        
        mock_post_resp = Mock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "objectId": mock_post_id,
            "extractionResultFile": {"url": mock_file_url, "name": "extraction_test.json.gz"}
        }
        
        mock_file_download_resp = Mock()
        mock_file_download_resp.status_code = 200
        mock_file_download_resp.content = compressed_data
        
        mock_fetch_client.get.side_effect = [mock_post_resp, mock_file_download_resp]
        
        # First call (store) uses mock_store_client, second call (fetch) uses mock_fetch_client
        mock_client_class.side_effect = [mock_store_client, mock_fetch_client]
        
        # Store
        result_id = await store_extraction_result_in_post(
            post_id=mock_post_id,
            structured_elements=test_elements,
            memory_requests=test_memories,
            element_summary=test_summary,
            decision="complex"
        )
        
        print(f"\n✅ Stored extraction: {result_id}")
        
        # Fetch
        fetched_data = await fetch_extraction_result_from_post(mock_post_id)
        
        assert fetched_data is not None
        assert fetched_data["decision"] == "complex"
        assert len(fetched_data["structured_elements"]) == 2
        assert len(fetched_data["memory_requests"]) == 2
        assert fetched_data["element_summary"] == test_summary
        
        print(f"✅ Fetched extraction successfully")
        print(f"   - Elements: {len(fetched_data['structured_elements'])}")
        print(f"   - Memories: {len(fetched_data['memory_requests'])}")
        print(f"   - Decision: {fetched_data['decision']}")


@pytest.mark.asyncio
async def test_parse_file_pydantic_usage():
    """Test that we're using ParseFile Pydantic model correctly"""
    from models.parse_server import ParseFile
    
    # Create ParseFile instance
    parse_file = ParseFile(
        type="File",
        name="test_extraction.json.gz",
        url="https://example.com/test_extraction.json.gz"
    )
    
    # Test serialization
    file_dict = parse_file.model_dump(exclude_none=True)
    
    assert file_dict["__type"] == "File"  # Should have __type
    assert file_dict["name"] == "test_extraction.json.gz"
    assert file_dict["url"] == "https://example.com/test_extraction.json.gz"
    
    print(f"\n✅ ParseFile Pydantic model works correctly")
    print(f"   Serialized: {file_dict}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

