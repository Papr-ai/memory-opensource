import asyncio
from cloud_plugins.temporal.activities.document_activities import (
    extract_structured_content_from_provider,
    generate_llm_optimized_memory_structures,
)
from models.shared_types import MemoryMetadata


def test_extract_structured_content_from_provider_tensorlake_minimal():
    provider_specific = {"content": "A page about data"}
    result = asyncio.run(extract_structured_content_from_provider(
        provider_specific=provider_specific,
        provider_name="tensorlake",
        base_metadata=MemoryMetadata(),
        organization_id="org1",
        namespace_id="ns1",
    ))
    assert "structured_elements" in result
    assert isinstance(result["structured_elements"], list)


def test_generate_llm_optimized_memory_structures_batch_shape():
    content_elements = [
        {"content_type": "text", "element_id": "e1", "content": "hello", "metadata": {}},
    ]
    result = asyncio.run(generate_llm_optimized_memory_structures(
        content_elements=content_elements,
        domain=None,
        base_metadata=MemoryMetadata(user_id="user123"),
        organization_id="org1",
        namespace_id="ns1",
        use_llm=False,
    ))
    assert "batch_request" in result
    assert "memory_requests" in result
    assert result["total_generated"] >= 1


