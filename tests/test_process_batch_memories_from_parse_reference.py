import asyncio
import types
import pytest


@pytest.mark.anyio
async def test_process_batch_memories_from_parse_reference_uses_batch_handler(monkeypatch):
    from cloud_plugins.temporal.activities import memory_activities as mem_act

    # 1) Mock fetch_batch_memories_from_parse to return two memories
    async def fake_fetch(post_id: str):
        return {
            "memories": [
                {"content": "A", "type": "text", "metadata": {"foo": "bar"}},
                {"content": "B", "type": "text", "metadata": {"foo": "baz"}},
            ]
        }

    # Patch at the source import (services.memory_management) because the activity
    # imports inside the function via `from services.memory_management import fetch_batch_memories_from_parse`
    monkeypatch.setattr(
        "services.memory_management.fetch_batch_memories_from_parse",
        fake_fetch,
        raising=False,
    )

    # 2) Stub a MemoryGraph singleton
    class DummyGraph:
        async def ensure_async_connection(self):
            return None

    async def fake_get_graph():
        return DummyGraph()

    monkeypatch.setattr(
        mem_act, "_get_memory_graph_singleton", fake_get_graph, raising=False
    )

    # 3) Mock common_add_memory_batch_handler to capture the batch_request
    captured = {}

    class DummyResult:
        status = "completed"
        total_processed = 2
        total_successful = 2
        total_failed = 0
        errors = []

    async def fake_batch_handler(
        request,
        memory_graph,
        background_tasks,
        auth_response,
        memory_request_batch,
        skip_background_processing,
        legacy_route,
    ):
        captured["batch_size"] = memory_request_batch.batch_size
        captured["mem_count"] = len(memory_request_batch.memories)
        return DummyResult()

    # Patch both the activity-local import target and the source in routes
    monkeypatch.setattr(
        "cloud_plugins.temporal.activities.memory_activities.common_add_memory_batch_handler",
        fake_batch_handler,
        raising=False,
    )
    monkeypatch.setattr(
        "routes.memory_routes.common_add_memory_batch_handler",
        fake_batch_handler,
        raising=False,
    )

    # 4) Execute under test
    res = await mem_act.process_batch_memories_from_parse_reference(
        post_id="post123",
        organization_id="org",
        namespace_id="ns",
        user_id="user",
        workspace_id="ws",
    )

    assert res["status"] in {"completed", "partial"}
    assert res["total_processed"] == 2
    # Verify we sent a single batch with both memories and capped batch_size
    assert captured["mem_count"] == 2
    assert 1 <= captured["batch_size"] <= 50


