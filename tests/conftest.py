import os
import certifi
# Ensure SSL cert env vars are set early to avoid httpx/ollama import issues
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
# Ensure protobuf uses pure-Python implementation before any imports that may load tokenizers/sentencepiece
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Override Docker service names with localhost for local testing (open-source only)
# This MUST happen before any modules are imported, so we do it at the top level
# When running tests locally, Docker service names won't resolve
# Only apply this override in open-source edition to avoid messing up cloud configuration
papr_edition = os.getenv("PAPR_EDITION", "opensource").lower()
if papr_edition == "opensource":
    parse_server_url = os.getenv("PARSE_SERVER_URL", "")
    if parse_server_url and "parse-server" in parse_server_url:
        # Replace Docker service name with localhost
        local_parse_url = parse_server_url.replace("parse-server", "localhost")
        # Remove trailing /parse if present, since the code adds /parse/classes
        # PARSE_SERVER_URL should be base URL (e.g., http://localhost:1337)
        if local_parse_url.endswith("/parse"):
            local_parse_url = local_parse_url[:-6]  # Remove trailing "/parse"
        elif local_parse_url.endswith("/parse/"):
            local_parse_url = local_parse_url[:-7]  # Remove trailing "/parse/"
        os.environ["PARSE_SERVER_URL"] = local_parse_url
    
    qdrant_url = os.getenv("QDRANT_URL", "")
    if qdrant_url and "qdrant" in qdrant_url and "localhost" not in qdrant_url:
        # Replace Docker service name with localhost (but preserve port if different)
        # qdrant:6333 -> localhost:6333
        local_qdrant_url = qdrant_url.replace("qdrant", "localhost")
        os.environ["QDRANT_URL"] = local_qdrant_url
    
    neo4j_url = os.getenv("NEO4J_URL", "")
    if neo4j_url and "neo4j:" in neo4j_url and "localhost" not in neo4j_url:
        # Replace Docker service name with localhost for bolt:// URLs
        # bolt://neo4j:7687 -> bolt://localhost:7687
        local_neo4j_url = neo4j_url.replace("neo4j:", "localhost:")
        os.environ["NEO4J_URL"] = local_neo4j_url

import asyncio
import contextlib
import pytest

try:
    from temporalio.worker import Worker  # type: ignore
    from cloud_plugins.temporal.client import get_temporal_client  # type: ignore
    from cloud_plugins.temporal.workflows.batch_memory import (
        ProcessBatchMemoryWorkflow,
        ProcessBatchMemoryFromRequestWorkflow,
    )  # type: ignore
    from cloud_plugins.temporal.activities import memory_activities  # type: ignore
    _TEMPORAL_AVAILABLE = True
except Exception:
    Worker = None  # type: ignore
    get_temporal_client = None  # type: ignore
    ProcessBatchMemoryWorkflow = None  # type: ignore
    ProcessBatchMemoryFromRequestWorkflow = None  # type: ignore
    memory_activities = None  # type: ignore
    _TEMPORAL_AVAILABLE = False


@pytest.fixture(scope="function")
async def temporal_worker():
    """Run a Temporal worker in-process for tests that need workflows/activities.

    Ensures webhook delivery works to the in-process FastAPI app via TEST_WEBHOOK_INPROC.
    """
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ.setdefault("TEST_WEBHOOK_INPROC", "true")

    if not _TEMPORAL_AVAILABLE:
        # No-op fixture when Temporal not available (most unit tests won't need it)
        yield
        return

    client = await get_temporal_client()
    from cloud_plugins.temporal.workflows.batch_memory import ProcessBatchMemoryFromPostWorkflow
    worker = Worker(
        client,
        task_queue="memory-processing",
        workflows=[
            ProcessBatchMemoryWorkflow,
            ProcessBatchMemoryFromRequestWorkflow,
            ProcessBatchMemoryFromPostWorkflow,
        ],
        activities=[
            # Memory processing activities - MUST MATCH start_all_workers.py
            memory_activities.add_memory_quick,
            memory_activities.batch_add_memory_quick,  # ✅ NEW: Batch version
            memory_activities.index_and_enrich_memory,
            memory_activities.update_relationships,
            memory_activities.process_memory_batch,
            memory_activities.send_webhook_notification,
            memory_activities.process_batch_memories_from_parse_reference,
            memory_activities.fetch_batch_memories_from_post,  # ✅ CRITICAL for document tests
            memory_activities.fetch_and_process_batch_request,
            # Fine-grained indexing activities
            memory_activities.idx_index_grouped_memory,
            memory_activities.idx_generate_graph_schema,
            memory_activities.idx_update_metrics,
            memory_activities.link_batch_memories_to_post,  # ✅ Links memories to Post objects
        ],
    )

    run_task = asyncio.create_task(worker.run())
    try:
        await asyncio.sleep(0.25)
        yield
    finally:
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task

import pytest
from unittest.mock import patch
try:
    import aiohttp  # type: ignore
    import ssl  # type: ignore
    import requests  # type: ignore
    import urllib3  # type: ignore
    _REQUESTS_AVAILABLE = True
except Exception:
    aiohttp = None  # type: ignore
    ssl = None  # type: ignore
    requests = None  # type: ignore
    urllib3 = None  # type: ignore
    _REQUESTS_AVAILABLE = False
try:
    from app_factory import create_app
except Exception:
    def create_app():  # minimal fallback to avoid import errors when FastAPI not installed
        class _Dummy:
            pass
        return _Dummy()

# Disable SSL verification warnings globally for testing (if requests stack present)
if _REQUESTS_AVAILABLE:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@pytest.fixture(autouse=True)
def disable_ssl_verification():
    """Disable SSL verification for all requests if requests is available."""
    if not _REQUESTS_AVAILABLE:
        # No-op when requests stack is not available in the test env
        yield
        return

    old_post = requests.post
    old_get = requests.get

    def patched_post(*args, **kwargs):
        kwargs['verify'] = False
        return old_post(*args, **kwargs)

    def patched_get(*args, **kwargs):
        kwargs['verify'] = False
        return old_get(*args, **kwargs)

    # Create a custom SSL context that doesn't verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Patch requests
    with patch('requests.post', side_effect=patched_post), \
         patch('requests.get', side_effect=patched_get):
        yield

@pytest.fixture
async def app():
    # Create a fresh app (and MemoryGraph) per test to avoid sharing asyncio primitives across loops
    # DON'T reset MongoDB singleton - let it persist across tests for better performance and stability
    # The singleton handles connection health checks and auto-reconnection
    # Only reset if the connection is actually unhealthy (handled by get_mongo_db)

    # Ensure MongoDB environment variables are set for tests BEFORE creating the app
    # This is critical because the MemoryGraph is created during app lifespan startup
    import os
    if not os.getenv("DATABASE_URI"):
        # Set DATABASE_URI from MONGO_URI if available, or use a default test connection
        mongo_uri = os.getenv("MONGO_URI")
        if mongo_uri:
            os.environ["DATABASE_URI"] = mongo_uri
            print(f"Set DATABASE_URI from MONGO_URI for test: {mongo_uri[:50]}...")
        else:
            # Fallback: use a test MongoDB connection string
            # This should be a local MongoDB or test instance
            test_db_uri = "mongodb://localhost:27017/parsedev"
            os.environ["DATABASE_URI"] = test_db_uri
            print(f"Set DATABASE_URI to test fallback: {test_db_uri}")
    
    # Note: Docker service name overrides are now done at the top of conftest.py
    # before any modules are imported, so they're already applied here

    app_instance = create_app()

    # Manually trigger lifespan startup for tests
    async with app_instance.router.lifespan_context(app_instance):
        yield app_instance


# MemoryGraph cleanup is handled by the app fixture's lifespan context
# No need for separate autouse fixture