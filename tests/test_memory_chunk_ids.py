"""
Test that memoryChunkIds are correctly populated for both single and multi-chunk memories.
"""
import pytest
import os
from fastapi import BackgroundTasks
from starlette.requests import Request as StarletteRequest
from models.memory_models import AddMemoryRequest, MemoryMetadata, OptimizedAuthResponse
from routes.memory_routes import common_add_memory_handler
from memory.memory_graph import MemoryGraph


@pytest.mark.asyncio
async def test_single_chunk_memory_has_chunk_ids():
    """Test that a short text memory has memoryChunkIds set to [memoryId_0]"""
    
    # Short content that will generate only 1 chunk
    short_content = "This is a short memory item that will only create one chunk."
    
    memory_request = AddMemoryRequest(
        content=short_content,
        type="text",
        metadata=MemoryMetadata(
            user_id=os.getenv("TEST_USER_ID", "mhnkVbAdgG"),
            workspace_id=os.getenv("TEST_WORKSPACE_ID"),
        )
    )
    
    # Build auth response
    auth = OptimizedAuthResponse(
        developer_id=os.getenv("TEST_DEVELOPER_ID", "mhnkVbAdgG"),
        end_user_id=os.getenv("TEST_USER_ID", "mhnkVbAdgG"),
        workspace_id=os.getenv("TEST_WORKSPACE_ID"),
        is_qwen_route=False,
        session_token=os.getenv("TEST_SESSION_TOKEN"),
        api_key=os.getenv("TEST_API_KEY")
    )
    
    # Build minimal request
    header_items = [
        (b"x-client-type", b"test_client"),
        (b"content-type", b"application/json")
    ]
    if os.getenv("TEST_API_KEY"):
        header_items.extend([
            (b"x-api-key", os.getenv("TEST_API_KEY").encode()),
            (b"authorization", f"APIKey {os.getenv('TEST_API_KEY')}".encode())
        ])
    
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": "/test-memory",
        "headers": header_items
    }
    
    async def _empty_receive():
        return {"type": "http.request"}
    
    request = StarletteRequest(scope, _empty_receive)
    
    # Get memory graph
    memory_graph = MemoryGraph()
    
    # Call handler with skip_background_processing=True for faster test
    result = await common_add_memory_handler(
        request=request,
        memory_graph=memory_graph,
        background_tasks=BackgroundTasks(),
        neo_session=None,
        auth_response=auth,
        memory_request=memory_request,
        skip_background_processing=True,
        legacy_route=True
    )
    
    # Verify result
    assert result.success, f"Memory creation failed: {result.error}"
    assert result.data, "No memory data returned"
    assert len(result.data) > 0, "Empty memory data list"
    
    first_memory = result.data[0]
    print(f"\n✅ Single-chunk memory created:")
    print(f"   memoryId: {first_memory.memoryId}")
    print(f"   memoryChunkIds: {first_memory.memoryChunkIds}")
    
    # Verify memoryChunkIds is set and has at least one entry
    assert first_memory.memoryChunkIds, "memoryChunkIds is empty!"
    assert len(first_memory.memoryChunkIds) >= 1, f"Expected at least 1 chunk ID, got {len(first_memory.memoryChunkIds)}"
    
    # For single chunk, verify format is [memoryId_0]
    expected_chunk_id = f"{first_memory.memoryId}_0"
    assert expected_chunk_id in first_memory.memoryChunkIds, \
        f"Expected chunk ID {expected_chunk_id} not found in {first_memory.memoryChunkIds}"


@pytest.mark.asyncio
async def test_multi_chunk_memory_has_all_chunk_ids():
    """Test that a long text memory has multiple chunk IDs in memoryChunkIds"""
    
    # Long content that will generate multiple chunks (each chunk is ~500 tokens)
    long_content = """
    This is a very long memory item that will create multiple chunks when processed.
    
    """ + "\n".join([
        f"Paragraph {i}: " + " ".join([
            "This is sentence number {j} in paragraph {i}. ".format(j=j, i=i) 
            for j in range(50)
        ])
        for i in range(20)  # 20 paragraphs with 50 sentences each = ~1000 sentences
    ])
    
    memory_request = AddMemoryRequest(
        content=long_content,
        type="text",
        metadata=MemoryMetadata(
            user_id=os.getenv("TEST_USER_ID", "mhnkVbAdgG"),
            workspace_id=os.getenv("TEST_WORKSPACE_ID"),
        )
    )
    
    # Build auth response
    auth = OptimizedAuthResponse(
        developer_id=os.getenv("TEST_DEVELOPER_ID", "mhnkVbAdgG"),
        end_user_id=os.getenv("TEST_USER_ID", "mhnkVbAdgG"),
        workspace_id=os.getenv("TEST_WORKSPACE_ID"),
        is_qwen_route=False,
        session_token=os.getenv("TEST_SESSION_TOKEN"),
        api_key=os.getenv("TEST_API_KEY")
    )
    
    # Build minimal request
    header_items = [
        (b"x-client-type", b"test_client"),
        (b"content-type", b"application/json")
    ]
    if os.getenv("TEST_API_KEY"):
        header_items.extend([
            (b"x-api-key", os.getenv("TEST_API_KEY").encode()),
            (b"authorization", f"APIKey {os.getenv('TEST_API_KEY')}".encode())
        ])
    
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": "/test-memory",
        "headers": header_items
    }
    
    async def _empty_receive():
        return {"type": "http.request"}
    
    request = StarletteRequest(scope, _empty_receive)
    
    # Get memory graph
    memory_graph = MemoryGraph()
    
    # Call handler with skip_background_processing=True for faster test
    result = await common_add_memory_handler(
        request=request,
        memory_graph=memory_graph,
        background_tasks=BackgroundTasks(),
        neo_session=None,
        auth_response=auth,
        memory_request=memory_request,
        skip_background_processing=True,
        legacy_route=True
    )
    
    # Verify result
    assert result.success, f"Memory creation failed: {result.error}"
    assert result.data, "No memory data returned"
    assert len(result.data) > 0, "Empty memory data list"
    
    first_memory = result.data[0]
    print(f"\n✅ Multi-chunk memory created:")
    print(f"   memoryId: {first_memory.memoryId}")
    print(f"   memoryChunkIds: {first_memory.memoryChunkIds}")
    print(f"   Number of chunks: {len(first_memory.memoryChunkIds)}")
    
    # Verify memoryChunkIds is set and has multiple entries
    assert first_memory.memoryChunkIds, "memoryChunkIds is empty!"
    assert len(first_memory.memoryChunkIds) > 1, \
        f"Expected multiple chunk IDs for long text, got only {len(first_memory.memoryChunkIds)}"
    
    # Verify all chunk IDs follow the format memoryId_0, memoryId_1, etc.
    base_id = first_memory.memoryId
    for idx, chunk_id in enumerate(first_memory.memoryChunkIds):
        expected_chunk_id = f"{base_id}_{idx}"
        assert chunk_id == expected_chunk_id, \
            f"Chunk {idx} has incorrect ID. Expected {expected_chunk_id}, got {chunk_id}"
    
    print(f"   ✅ All {len(first_memory.memoryChunkIds)} chunk IDs are correctly formatted")


if __name__ == "__main__":
    import asyncio
    
    print("=" * 80)
    print("Testing single-chunk memory...")
    print("=" * 80)
    asyncio.run(test_single_chunk_memory_has_chunk_ids())
    
    print("\n" + "=" * 80)
    print("Testing multi-chunk memory...")
    print("=" * 80)
    asyncio.run(test_multi_chunk_memory_has_all_chunk_ids())
    
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)

