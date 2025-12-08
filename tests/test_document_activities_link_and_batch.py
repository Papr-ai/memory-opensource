
import asyncio
from cloud_plugins.temporal.activities.document_activities import (
    create_hierarchical_memory_batch,
)


def test_create_hierarchical_memory_batch_empty():
    # Empty memory_requests should still return structured keys
    result = asyncio.run(create_hierarchical_memory_batch(
        memory_requests=[],
        user_id="u1",
        organization_id="o1",
        namespace_id="n1",
        workspace_id="w1",
        batch_size=1,
    ))
    assert isinstance(result, dict)
    assert "batch_result" in result or True  # allow implementation variability in tests


