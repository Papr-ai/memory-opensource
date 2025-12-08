"""
Unit tests for extract_structured_content_from_provider activity
to debug why we're only getting summaries instead of full content
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from models.shared_types import MemoryMetadata
from models.hierarchical_models import ContentElement, TextElement


@pytest.fixture
def sample_reducto_response():
    """Sample Reducto response structure (based on actual API response)"""
    return {
        "result": {
            "chunks": [
                {
                    "blocks": [
                        {
                            "type": "text",
                            "content": "This is the first paragraph of actual content from the document.",
                            "confidence": 0.95
                        },
                        {
                            "type": "text",
                            "content": "This is the second paragraph with more detailed information.",
                            "confidence": 0.92
                        }
                    ]
                },
                {
                    "blocks": [
                        {
                            "type": "table",
                            "content": "Q1\t100\t200\nQ2\t150\t250",
                            "confidence": 0.88
                        }
                    ]
                },
                {
                    "blocks": [
                        {
                            "type": "text",
                            "content": "This is content from page 3 with important details.",
                            "confidence": 0.94
                        }
                    ]
                }
            ]
        },
        "usage": {
            "num_pages": 30,
            "credits": 30.0
        }
    }


@pytest.fixture
def base_metadata():
    """Base metadata for memory requests"""
    return MemoryMetadata(
        organization_id="test_org",
        namespace_id="test_namespace",
        user_id="test_user",
        customMetadata={
            "test_id": "test_extract_structured",
            "source": "unit_test"
        }
    )


@pytest.mark.asyncio
async def test_extract_structured_content_returns_full_content(sample_reducto_response, base_metadata):
    """Test that extract_structured_content_from_provider returns FULL content, not just summary"""
    
    # Import the activity
    from cloud_plugins.temporal.activities.document_activities import extract_structured_content_from_provider
    
    # Call the activity directly (not via Temporal)
    result = await extract_structured_content_from_provider(
        provider_specific=sample_reducto_response,
        provider_name="reducto",
        base_metadata=base_metadata,
        organization_id="test_org",
        namespace_id="test_namespace"
    )
    
    # Assertions
    assert result is not None, "Result should not be None"
    assert "decision" in result, "Result should include decision field"
    assert "structured_elements" in result, "Result should include structured_elements"
    assert "memory_requests" in result, "Result should include memory_requests"
    
    # Check decision
    decision = result["decision"]
    print(f"\n=== DECISION: {decision} ===")
    
    # Get the content elements/memory requests
    structured_elements = result.get("structured_elements", [])
    memory_requests = result.get("memory_requests", [])
    
    print(f"\n=== STRUCTURED ELEMENTS: {len(structured_elements)} ===")
    for i, elem in enumerate(structured_elements[:3]):  # Print first 3
        content_preview = elem.get("content", "")[:100] if isinstance(elem, dict) else str(elem)[:100]
        print(f"Element {i}: {content_preview}...")
    
    print(f"\n=== MEMORY REQUESTS: {len(memory_requests)} ===")
    for i, mem in enumerate(memory_requests[:3]):  # Print first 3
        content = mem.get("content", "") if isinstance(mem, dict) else ""
        content_preview = content[:100] if content else "NO CONTENT"
        print(f"Memory {i}: {content_preview}...")
    
    # THE KEY ASSERTION: We should have MORE than just 1 summary element
    total_items = len(structured_elements) + len(memory_requests)
    assert total_items > 1, f"Expected multiple content items, got only {total_items}. This suggests we're only getting a summary!"
    
    # Check that we're NOT just getting a document summary
    if memory_requests:
        first_memory_content = memory_requests[0].get("content", "") if isinstance(memory_requests[0], dict) else ""
        assert "Document Summary:" not in first_memory_content or len(memory_requests) > 1, \
            "Should have more than just a document summary!"
    
    # Verify we extracted actual content from blocks
    all_content = []
    for mem in memory_requests:
        content = mem.get("content", "") if isinstance(mem, dict) else ""
        all_content.append(content)
    
    combined_content = " ".join(all_content)
    
    # These phrases are from our sample blocks - they should appear in the extracted content
    assert "first paragraph of actual content" in combined_content or len(memory_requests) >= 3, \
        "Should extract actual block content, not just summary!"
    
    print(f"\n✅ TEST PASSED: Extracted {total_items} items (not just a summary)")


@pytest.mark.asyncio
async def test_reducto_transformer_directly(sample_reducto_response, base_metadata):
    """Test reducto_memory_transformer directly to see what it returns"""
    
    from core.document_processing.reducto_memory_transformer import transform_reducto_to_memories
    
    # Call the transformer directly
    memory_requests = transform_reducto_to_memories(
        reducto_response=sample_reducto_response,
        base_metadata=base_metadata.model_dump(),
        organization_id="test_org",
        namespace_id="test_namespace",
        user_id="test_user"
    )
    
    print(f"\n=== REDUCTO TRANSFORMER RETURNED {len(memory_requests)} MEMORIES ===")
    
    for i, mem in enumerate(memory_requests[:5]):  # Print first 5
        content = mem.content if hasattr(mem, 'content') else "NO CONTENT"
        content_preview = content[:150] if content else "EMPTY"
        mem_type = mem.metadata.customMetadata.get('content_type') if mem.metadata and mem.metadata.customMetadata else 'unknown'
        print(f"\nMemory {i} (type: {mem_type}):")
        print(f"  {content_preview}...")
    
    # We should get multiple memories from the chunks
    assert len(memory_requests) >= 3, f"Expected at least 3 memories (1 summary + 2 chunks), got {len(memory_requests)}"
    
    # Check content types
    content_types = []
    for mem in memory_requests:
        if mem.metadata and mem.metadata.customMetadata:
            content_types.append(mem.metadata.customMetadata.get('content_type', 'unknown'))
    
    print(f"\nContent types: {content_types}")
    
    # Should have different content types (summary, text, table, etc.)
    assert len(set(content_types)) > 1, f"Should have multiple content types, got: {set(content_types)}"
    
    print(f"\n✅ TRANSFORMER TEST PASSED: Got {len(memory_requests)} memories with {len(set(content_types))} different types")


@pytest.mark.asyncio
async def test_provider_adapter_extract_elements(sample_reducto_response, base_metadata):
    """Test provider_adapter.extract_structured_elements to see what it returns"""
    
    from core.document_processing.provider_adapter import extract_structured_elements
    
    # Call the adapter
    elements = extract_structured_elements(
        provider_name="reducto",
        provider_specific=sample_reducto_response,
        base_metadata=base_metadata.model_dump(),
        organization_id="test_org",
        namespace_id="test_namespace"
    )
    
    print(f"\n=== PROVIDER ADAPTER RETURNED {len(elements)} ELEMENTS ===")
    
    for i, elem in enumerate(elements[:5]):  # Print first 5
        elem_type = elem.content_type.value if hasattr(elem, 'content_type') else 'unknown'
        content = elem.content if hasattr(elem, 'content') else "NO CONTENT"
        content_preview = content[:150] if content else "EMPTY"
        print(f"\nElement {i} (type: {elem_type}):")
        print(f"  {content_preview}...")
    
    # Should get multiple elements
    assert len(elements) >= 3, f"Expected at least 3 elements, got {len(elements)}"
    
    # Check that elements are proper ContentElement objects
    for elem in elements:
        assert hasattr(elem, 'content_type'), "Element should have content_type"
        assert hasattr(elem, 'content'), "Element should have content"
        assert hasattr(elem, 'element_id'), "Element should have element_id"
    
    print(f"\n✅ ADAPTER TEST PASSED: Got {len(elements)} proper ContentElement objects")


@pytest.mark.asyncio  
async def test_with_mock_parse_post(sample_reducto_response, base_metadata):
    """Test the activity with a mocked Parse Post fetch (simulating post_id reference)"""
    
    from cloud_plugins.temporal.activities.document_activities import extract_structured_content_from_provider
    import httpx
    
    # Mock the Parse Server HTTP call
    with patch('httpx.AsyncClient') as mock_client:
        # Setup mock response for Parse Post fetch
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "objectId": "test_post_123",
            "content": {
                "provider": "reducto",
                "provider_result_file": {
                    "url": "https://example.com/provider_result.json"
                }
            },
            "processingMetadata": {
                "provider": "reducto",
                "total_pages": 30
            }
        }
        
        # Mock the provider result file download
        mock_file_response = Mock()
        mock_file_response.status_code = 200
        mock_file_response.content = json.dumps(sample_reducto_response).encode('utf-8')
        
        # Setup async context manager mock
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(side_effect=[mock_response, mock_file_response])
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        # Call activity with post_id reference
        result = await extract_structured_content_from_provider(
            provider_specific={"post": {"objectId": "test_post_123"}},
            provider_name="reducto",
            base_metadata=base_metadata,
            organization_id="test_org",
            namespace_id="test_namespace"
        )
        
        print(f"\n=== WITH PARSE POST MOCK ===")
        print(f"Decision: {result.get('decision')}")
        print(f"Structured elements: {len(result.get('structured_elements', []))}")
        print(f"Memory requests: {len(result.get('memory_requests', []))}")
        
        # Should have extracted full content
        total_items = len(result.get('structured_elements', [])) + len(result.get('memory_requests', []))
        assert total_items > 1, f"With Parse post fetch, should get full content, not just summary. Got {total_items} items"
        
        print(f"\n✅ PARSE POST TEST PASSED: Extracted {total_items} items from Parse Post reference")


if __name__ == "__main__":
    """Run tests directly with pytest"""
    pytest.main([__file__, "-v", "-s"])
