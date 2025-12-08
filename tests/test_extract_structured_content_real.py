"""
Test extract_structured_content_from_provider with real Reducto JSON file
"""
import pytest
import json
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from os import environ as env
from models.memory_models import AddMemoryRequest
from models.shared_types import MemoryMetadata

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)


@pytest.fixture
def real_reducto_response():
    """Load the real Reducto JSON file"""
    json_path = Path("/Users/shawkatkabbara/Downloads/b1ee8b3479b29f40964bdaa830163b19_provider_result_f8141f7d-88ba-4145-925c-0b025b22d6c7.json")
    
    if not json_path.exists():
        pytest.skip(f"Real Reducto JSON file not found at {json_path}")
    
    with open(json_path, 'r') as f:
        return json.load(f)


@pytest.fixture
def base_metadata():
    """Create base metadata for testing using real env variables"""
    return MemoryMetadata(
        organization_id=env.get("TEST_ORGANIZATION_ID", "Ky6jxP0yxI"),
        namespace_id=env.get("TEST_NAMESPACE_ID", "MwnkcNiGZU"),
        user_id=env.get("TEST_USER_ID", "7hd4717pdV"),
        external_user_id=env.get("TEST_EXTERNAL_USER_ID"),
        workspace_id=env.get("TEST_WORKSPACE_ID", "pohYfXWoOK"),
        customMetadata={
            "test_id": "real_reducto_test",
            "source": "pytest_real_file_test"
        }
    )


@pytest.mark.asyncio
async def test_extract_with_real_reducto_file(real_reducto_response, base_metadata):
    """Test extraction with real Reducto JSON file"""
    from cloud_plugins.temporal.activities.document_activities import extract_structured_content_from_provider
    
    print(f"\n=== USING REAL ENV VARIABLES ===")
    print(f"Organization ID: {base_metadata.organization_id}")
    print(f"Namespace ID: {base_metadata.namespace_id}")
    print(f"User ID: {base_metadata.user_id}")
    print(f"Workspace ID: {base_metadata.workspace_id}")
    
    result = await extract_structured_content_from_provider(
        provider_specific=real_reducto_response,
        provider_name="reducto",
        base_metadata=base_metadata,
        organization_id=base_metadata.organization_id,
        namespace_id=base_metadata.namespace_id
    )
    
    print(f"\n{'='*80}")
    print(f"FULL RETURN FROM extract_structured_content_from_provider")
    print(f"{'='*80}")
    
    import json
    # Pretty print the full result
    result_for_print = {
        "decision": result.get("decision"),
        "provider": result.get("provider"),
        "structure_analysis": result.get("structure_analysis"),
        "element_summary": result.get("element_summary"),
        "structured_elements_count": len(result.get("structured_elements", [])),
        "memory_requests_count": len(result.get("memory_requests", [])),
        "structured_elements": [
            {
                "element_id": getattr(elem, "element_id", None),
                "content_type": str(getattr(elem, "content_type", None)),
                "content_preview": (getattr(elem, "content", "")[:200] + "...") if hasattr(elem, "content") and len(getattr(elem, "content", "")) > 200 else getattr(elem, "content", ""),
                "metadata": getattr(elem, "metadata", None)
            }
            for elem in result.get("structured_elements", [])[:3]  # First 3 only
        ],
        "memory_requests": [
            {
                "content_preview": (mem.content[:200] + "...") if hasattr(mem, "content") and len(mem.content) > 200 else (mem.content if hasattr(mem, "content") else str(mem)[:200]),
                "type": str(mem.type) if hasattr(mem, "type") else None,
                "metadata": mem.metadata.model_dump() if hasattr(mem, "metadata") else None
            }
            for mem in result.get("memory_requests", [])[:3]  # First 3 only
        ]
    }
    
    print(json.dumps(result_for_print, indent=2, default=str))
    print(f"{'='*80}\n")
    
    print(f"\n=== REAL REDUCTO FILE TEST ===")
    print(f"Decision: {result['decision']}")
    print(f"Structured elements: {len(result['structured_elements'])}")
    print(f"Memory requests: {len(result['memory_requests'])}")
    print(f"Provider: {result['provider']}")
    print(f"Structure analysis: {result['structure_analysis']}")
    
    # Assertions
    assert result["decision"] in ["simple", "complex"]
    assert result["provider"] == "reducto"
    
    # Should have structured elements or memory requests
    total_items = len(result["structured_elements"]) + len(result["memory_requests"])
    assert total_items > 0, "Should have extracted some content"
    
    print(f"\n=== CONTENT SUMMARY ===")
    print(f"Total extracted items: {total_items}")
    
    # Show first few items
    if result["structured_elements"]:
        print(f"\n=== FIRST 3 STRUCTURED ELEMENTS ===")
        for i, elem in enumerate(result["structured_elements"][:3]):
            content_preview = elem.content[:100] if hasattr(elem, 'content') else str(elem)[:100]
            print(f"Element {i}: {content_preview}...")
    
    if result["memory_requests"]:
        print(f"\n=== FIRST 3 MEMORY REQUESTS ===")
        for i, mem in enumerate(result["memory_requests"][:3]):
            if isinstance(mem, dict):
                content_preview = mem.get("content", "")[:100]
            else:
                content_preview = mem.content[:100] if hasattr(mem, 'content') else str(mem)[:100]
            print(f"Memory {i}: {content_preview}...")
    
    print(f"\n✅ TEST PASSED: Successfully processed real Reducto file with {total_items} items")


@pytest.mark.asyncio
async def test_provider_type_parser_with_real_file(real_reducto_response):
    """Test the provider type parser with real Reducto file"""
    from core.document_processing.provider_type_parser import parse_with_provider_sdk
    
    # Test typed parsing
    parsed = parse_with_provider_sdk("reducto", real_reducto_response)
    
    assert isinstance(parsed, dict)
    assert "result" in parsed or "job_id" in parsed
    
    print(f"\n=== PROVIDER TYPE PARSER TEST ===")
    print(f"Parsed successfully: {isinstance(parsed, dict)}")
    print(f"Has result: {'result' in parsed}")
    print(f"Has job_id: {'job_id' in parsed}")
    
    if "result" in parsed:
        result = parsed["result"]
        if isinstance(result, dict) and "parse" in result:
            parse_result = result["parse"]["result"]
            chunks = parse_result.get("chunks", [])
            print(f"Number of chunks: {len(chunks)}")
            
            if chunks:
                first_chunk = chunks[0]
                blocks = first_chunk.get("blocks", [])
                print(f"Blocks in first chunk: {len(blocks)}")
                
                if blocks:
                    print(f"First block type: {blocks[0].get('type')}")
                    print(f"First block content preview: {blocks[0].get('content', '')[:100]}...")
    
    print(f"✅ TYPE PARSER TEST PASSED")


@pytest.mark.asyncio
async def test_reducto_transformer_with_real_file(real_reducto_response, base_metadata):
    """Test Reducto transformer directly with real file"""
    from core.document_processing.reducto_memory_transformer import transform_reducto_to_memories
    
    memories = transform_reducto_to_memories(
        reducto_response=real_reducto_response,
        base_metadata=base_metadata.model_dump(),
        organization_id=base_metadata.organization_id,
        namespace_id=base_metadata.namespace_id,
        user_id=base_metadata.user_id
    )
    
    print(f"\n=== REDUCTO TRANSFORMER TEST (REAL FILE) ===")
    print(f"Generated {len(memories)} memory objects")
    
    # Check memory types
    content_types = {}
    for mem in memories:
        ct = mem.metadata.customMetadata.get("content_type", "unknown")
        content_types[ct] = content_types.get(ct, 0) + 1
    
    print(f"Content type breakdown: {content_types}")
    
    # Show first few memories
    print(f"\n=== FIRST 3 MEMORIES ===")
    for i, mem in enumerate(memories[:3]):
        print(f"Memory {i} (type: {mem.metadata.customMetadata.get('content_type')}): {mem.content[:150]}...")
    
    assert len(memories) > 0, "Should generate at least one memory"
    assert any(m.metadata.customMetadata.get("content_type") == "document_summary" for m in memories), "Should have document summary"
    
    print(f"\n✅ TRANSFORMER TEST PASSED: Generated {len(memories)} memories")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

