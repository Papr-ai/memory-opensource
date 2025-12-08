"""
Add an engineering learning note to Memory via the existing batch handler path.
Usage:
  python -m scripts.add_agent_learning "Title" "Body text..." --user-id <id> --api-key <key>
This uses customMetadata to mark the memory as an agent learning for the memory server.
"""

import argparse
import asyncio
import os
from typing import Optional
from models.memory_models import AddMemoryRequest, MemoryType
from models.shared_types import MemoryMetadata
from services.logger_singleton import LoggerSingleton
from memory.memory_graph import MemoryGraph
from routes.memory_routes import common_add_memory_batch_handler
from models.memory_models import BatchMemoryRequest
from models.parse_server import AddMemoryResponse
from fastapi import BackgroundTasks
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

logger = LoggerSingleton.get_logger(__name__)


async def add_learning(title: str, body: str, user_id: str, api_key: Optional[str] = None) -> AddMemoryResponse:
    memory_graph = MemoryGraph()
    await memory_graph.ensure_async_connection()

    metadata = MemoryMetadata(
        topics=["engineering", "learning", "agent"],
        customMetadata={
            "category": "memory_server_eng_learnings",
            "title": title,
            "source": "agent.md",
        }
    )

    add_req = AddMemoryRequest(
        content=body,
        type=MemoryType.TEXT,
        metadata=metadata,
    )

    batch = BatchMemoryRequest(memories=[add_req], batch_size=1)

    # Minimal mock FastAPI request for handler contract
    # Use provided api_key, fallback to TEST_X_USER_API_KEY from env, or raise error
    final_api_key = api_key or os.getenv("TEST_X_USER_API_KEY")
    if not final_api_key:
        raise ValueError("API key required: provide --api-key argument or set TEST_X_USER_API_KEY environment variable")
    
    class _Req:
        headers = {"X-Client-Type": "papr_plugin", "X-API-Key": final_api_key}

    result = await common_add_memory_batch_handler(
        request=_Req(),
        memory_graph=memory_graph,
        background_tasks=BackgroundTasks(),
        auth_response=None,
        memory_request_batch=batch,
        skip_background_processing=True,
        legacy_route=False
    )

    try:
        await memory_graph.cleanup()
    except Exception:
        pass

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("title")
    parser.add_argument("body")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--api-key", required=False)
    args = parser.parse_args()

    resp = asyncio.run(add_learning(args.title, args.body, args.user_id, args.api_key))
    print(getattr(resp, "status", "unknown"), getattr(resp, "code", 0))


if __name__ == "__main__":
    main()


