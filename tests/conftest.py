import os
import certifi
# Ensure SSL cert env vars are set early to avoid httpx/ollama import issues
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
# Ensure protobuf uses pure-Python implementation before any imports that may load tokenizers/sentencepiece
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
# Prevent OpenBLAS/OpenMP thread conflict that causes hangs with local embedding models
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ============================================
# Docker vs Local Environment Detection
# ============================================
# When running tests INSIDE Docker (e.g. via docker compose run test-runner),
# services are reachable by Docker service name (neo4j, qdrant, mongodb, etc.).
# When running tests LOCALLY, services are reachable at localhost.

def is_running_in_docker():
    """Check if we're running inside a Docker container."""
    if os.path.exists('/.dockerenv'):
        return True
    try:
        with open('/proc/1/cgroup', 'r') as f:
            return 'docker' in f.read()
    except Exception:
        return False

IN_DOCKER = is_running_in_docker()
papr_edition = os.getenv("PAPR_EDITION", "opensource").lower()

if IN_DOCKER:
    # INSIDE Docker: Docker Compose already sets the correct service names
    # (neo4j, qdrant, mongodb, etc.) via the `environment:` section.
    #
    # The problem: test files and app modules call load_dotenv() which reads
    # the baked-in .env file containing "localhost" URLs, overriding the
    # correct Docker service names.
    #
    # The fix: Disable dotenv loading entirely. Docker Compose env vars are
    # already correct, so we don't need .env files at all.
    os.environ["USE_DOTENV"] = "false"

    # Also force-set the critical URLs in case something slipped through
    _DOCKER_ENV_OVERRIDES = {
        "NEO4J_URL": "bolt://neo4j:7687",
        "QDRANT_URL": "http://qdrant:6333",
        "PARSE_SERVER_URL": "http://parse-server:1337",
        "MONGO_URI": os.getenv("MONGO_URI", "mongodb://admin:password@mongodb:27017/papr_memory?authSource=admin"),
        "DATABASE_URI": os.getenv("DATABASE_URI", "mongodb://admin:password@mongodb:27017/papr_memory?authSource=admin"),
        "REDIS_URL": os.getenv("REDIS_URL", "redis://:password@redis:6379/0"),
    }
    for key, val in _DOCKER_ENV_OVERRIDES.items():
        os.environ[key] = val

elif papr_edition == "opensource":
    # OUTSIDE Docker (local development): Replace Docker service names with localhost
    parse_server_url = os.getenv("PARSE_SERVER_URL", "")
    if parse_server_url and "parse-server" in parse_server_url:
        local_parse_url = parse_server_url.replace("parse-server", "localhost")
        if local_parse_url.endswith("/parse"):
            local_parse_url = local_parse_url[:-6]
        elif local_parse_url.endswith("/parse/"):
            local_parse_url = local_parse_url[:-7]
        os.environ["PARSE_SERVER_URL"] = local_parse_url

    qdrant_url = os.getenv("QDRANT_URL", "")
    if qdrant_url and "qdrant" in qdrant_url and "localhost" not in qdrant_url:
        os.environ["QDRANT_URL"] = qdrant_url.replace("qdrant", "localhost")

    neo4j_url = os.getenv("NEO4J_URL", "")
    if neo4j_url and "neo4j:" in neo4j_url and "localhost" not in neo4j_url:
        os.environ["NEO4J_URL"] = neo4j_url.replace("neo4j:", "localhost:")

import asyncio
import contextlib
import pytest
import logging

# Set up logging for conftest
_conftest_logger = logging.getLogger(__name__)

# ============================================
# Auto-fetch test credentials from Parse Server
# ============================================
# This runs once at the start of the test session and fetches the test user
# credentials directly from MongoDB. This eliminates the need for hardcoded
# test credentials and ensures tests always use the correct values.

def _fetch_test_credentials_from_mongodb():
    """
    Fetch test user credentials from MongoDB.
    
    This queries the database directly for the test user created by the
    bootstrap script and returns all the credentials needed for tests.
    
    Returns:
        dict with TEST_USER_ID, TEST_SESSION_TOKEN, TEST_X_USER_API_KEY,
        TEST_ORGANIZATION_ID, TEST_NAMESPACE_ID, TEST_WORKSPACE_ID, TEST_TENANT_ID
        or None if fetch fails
    """
    try:
        from pymongo import MongoClient
        
        mongo_uri = os.getenv("MONGO_URI") or os.getenv("DATABASE_URI")
        if not mongo_uri:
            # Default for Docker environment
            mongo_uri = "mongodb://admin:password@mongodb:27017/papr_memory?authSource=admin"
        
        _conftest_logger.info(f"🔑 Fetching test credentials from MongoDB...")
        
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client.get_default_database()
        
        # Find the test user (created by bootstrap script)
        user = db.get_collection("_User").find_one({"email": "test@papr.ai"})
        if not user:
            _conftest_logger.warning("⚠️ Test user 'test@papr.ai' not found in database")
            _conftest_logger.warning("   Run the bootstrap script first: scripts/bootstrap_opensource.py")
            return None
        
        user_id = user.get("_id")
        api_key = user.get("userAPIkey")
        
        # Get session token
        session = db.get_collection("_Session").find_one({"_p_user": f"_User${user_id}"})
        session_token = session.get("_session_token") if session else None
        
        # Get organization (from user's _p_organization pointer)
        org_id = None
        org_pointer = user.get("_p_organization", "")
        if org_pointer and org_pointer.startswith("Organization$"):
            org_id = org_pointer.replace("Organization$", "")
        
        # Get namespace (from organization's default_namespace or from APIKey)
        namespace_id = None
        if org_id:
            org_doc = db.Organization.find_one({"_id": org_id})
            if org_doc:
                ns_pointer = org_doc.get("_p_default_namespace", "")
                if ns_pointer and ns_pointer.startswith("Namespace$"):
                    namespace_id = ns_pointer.replace("Namespace$", "")
        
        # Fallback: get namespace from APIKey
        if not namespace_id and api_key:
            api_key_doc = db.APIKey.find_one({"key": api_key})
            if api_key_doc:
                ns_pointer = api_key_doc.get("_p_namespace", "")
                if ns_pointer and ns_pointer.startswith("Namespace$"):
                    namespace_id = ns_pointer.replace("Namespace$", "")
                # Also check namespace_id field directly
                if not namespace_id:
                    namespace_id = api_key_doc.get("namespace_id")
        
        # Get workspace from workspace_follower
        workspace_id = None
        ws_follower_pointer = user.get("_p_isSelectedWorkspaceFollower", "")
        if ws_follower_pointer and ws_follower_pointer.startswith("workspace_follower$"):
            follower_id = ws_follower_pointer.replace("workspace_follower$", "")
            follower_doc = db.workspace_follower.find_one({"_id": follower_id})
            if follower_doc:
                ws_pointer = follower_doc.get("_p_workspace", "")
                if ws_pointer and ws_pointer.startswith("WorkSpace$"):
                    workspace_id = ws_pointer.replace("WorkSpace$", "")
        
        client.close()
        
        credentials = {
            "TEST_USER_ID": user_id,
            "TEST_SESSION_TOKEN": session_token,
            "TEST_X_USER_API_KEY": api_key,
            "TEST_ORGANIZATION_ID": org_id,
            "TEST_NAMESPACE_ID": namespace_id,
            "TEST_WORKSPACE_ID": workspace_id,
            "TEST_TENANT_ID": workspace_id,  # Tenant ID is same as workspace ID
        }
        
        _conftest_logger.info(f"✅ Test credentials fetched successfully:")
        _conftest_logger.info(f"   User ID: {user_id}")
        _conftest_logger.info(f"   Organization ID: {org_id}")
        _conftest_logger.info(f"   Namespace ID: {namespace_id}")
        _conftest_logger.info(f"   Workspace ID: {workspace_id}")
        
        return credentials
        
    except Exception as e:
        _conftest_logger.error(f"❌ Failed to fetch test credentials: {e}")
        return None


def _set_test_credentials_in_env():
    """
    Fetch test credentials and set them as environment variables.
    
    This is called once at module load time to ensure credentials are
    available before any tests run.
    
    IMPORTANT: Always fetch from MongoDB when running in Docker to ensure
    credentials match the actual database state.
    """
    # When running in Docker, ALWAYS fetch from MongoDB to get the correct values
    # This ensures credentials match the actual test user created by bootstrap
    if IN_DOCKER:
        _conftest_logger.info("🔑 Running in Docker - fetching credentials from MongoDB...")
        credentials = _fetch_test_credentials_from_mongodb()
        if credentials:
            for key, value in credentials.items():
                if value:
                    os.environ[key] = str(value)
                    _conftest_logger.info(f"   Set {key}={value}")
            return
        else:
            _conftest_logger.warning("⚠️ Could not auto-fetch test credentials from MongoDB")
            _conftest_logger.warning("   Tests requiring authentication may fail")
            _conftest_logger.warning("   Ensure the bootstrap script has been run")
            return
    
    # For local development, skip if credentials are already set from .env
    if os.getenv("TEST_X_USER_API_KEY") and os.getenv("TEST_ORGANIZATION_ID"):
        _conftest_logger.info("🔑 Test credentials already set in environment (local dev mode)")
        return
    
    # Fallback: try to fetch from MongoDB for local dev too
    credentials = _fetch_test_credentials_from_mongodb()
    if credentials:
        for key, value in credentials.items():
            if value:
                os.environ[key] = str(value)
                _conftest_logger.debug(f"   Set {key}={value[:20] if len(str(value)) > 20 else value}...")
    else:
        _conftest_logger.warning("⚠️ Could not auto-fetch test credentials")
        _conftest_logger.warning("   Tests requiring authentication may fail")
        _conftest_logger.warning("   Ensure the bootstrap script has been run")


# Auto-fetch credentials when conftest.py is loaded
# This happens before any tests run
_set_test_credentials_in_env()


@pytest.fixture(scope="session")
def test_credentials():
    """
    Provide test credentials as a fixture.
    
    This returns a dict with all the test credentials, either from
    environment variables (if set) or by auto-fetching from Parse Server.
    
    Usage in tests:
        def test_something(test_credentials):
            org_id = test_credentials["TEST_ORGANIZATION_ID"]
            api_key = test_credentials["TEST_X_USER_API_KEY"]
    """
    # Try to get from environment first (already set by _set_test_credentials_in_env)
    credentials = {
        "TEST_USER_ID": os.environ.get("TEST_USER_ID"),
        "TEST_SESSION_TOKEN": os.environ.get("TEST_SESSION_TOKEN"),
        "TEST_X_USER_API_KEY": os.environ.get("TEST_X_USER_API_KEY"),
        "TEST_ORGANIZATION_ID": os.environ.get("TEST_ORGANIZATION_ID"),
        "TEST_NAMESPACE_ID": os.environ.get("TEST_NAMESPACE_ID"),
        "TEST_WORKSPACE_ID": os.environ.get("TEST_WORKSPACE_ID"),
        "TEST_TENANT_ID": os.environ.get("TEST_TENANT_ID"),
    }
    
    # Verify we have the essential credentials
    if not credentials.get("TEST_X_USER_API_KEY"):
        pytest.skip("Test credentials not available. Ensure the bootstrap script has been run.")
    
    return credentials


# ============================================
# Session-scoped fixtures for performance
# ============================================

@pytest.fixture(scope="session")
def preload_embedding_model():
    """
    Load the embedding model once at the start of the test session.
    
    This prevents the model from being loaded multiple times during tests,
    which causes memory issues and segmentation faults.
    
    Scope: session - loaded once per test run
    """
    import os
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
    
    if not use_local:
        # Using cloud embeddings - no model to preload
        yield None
        return
    
    try:
        from models.embedding_model import EmbeddingModel
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("=" * 60)
        logger.info("🔄 PRE-LOADING EMBEDDING MODEL FOR TEST SESSION")
        logger.info("=" * 60)
        
        # Create a singleton instance - this will load the model once
        embedding_model = EmbeddingModel()
        
        # Verify the model loaded successfully
        if hasattr(EmbeddingModel, '_qwen0pt6b_model_instance') and \
           EmbeddingModel._qwen0pt6b_model_instance is not None:
            logger.info("✅ Local embedding model (Qwen3-0.6B) loaded successfully")
            logger.info("✅ Model will be reused across all tests")
        else:
            logger.warning("⚠️  Local embedding model not loaded - will use cloud API")
        
        logger.info("=" * 60)
        
        yield embedding_model
        
        logger.info("🧹 Test session complete - embedding model will be garbage collected")
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Failed to preload embedding model: {e}")
        logger.warning("⚠️  Tests will attempt to load model on-demand (may cause issues)")
        yield None

@pytest.fixture(scope="session")
def event_loop_policy():
    """
    Set the event loop policy for the test session.
    
    This ensures consistent async behavior across all tests.
    """
    import asyncio
    policy = asyncio.get_event_loop_policy()
    yield policy
    # Cleanup is handled by pytest-asyncio

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
async def app(preload_embedding_model):
    """
    Create FastAPI app instance for testing.
    
    Now uses preload_embedding_model fixture to ensure the embedding model
    is loaded once per session and reused across all tests.
    
    Args:
        preload_embedding_model: Session-scoped fixture that pre-loads the model
    """
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
    
    # The preload_embedding_model fixture ensures the model is already loaded
    # The EmbeddingModel class will use the singleton instance automatically

    app_instance = create_app()

    # Manually trigger lifespan startup for tests
    async with app_instance.router.lifespan_context(app_instance):
        yield app_instance


# MemoryGraph cleanup is handled by the app fixture's lifespan context
# No need for separate autouse fixture