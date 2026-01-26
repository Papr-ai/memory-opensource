import os
import sys
import secrets
import asyncio
import json
import yaml
import httpx
import time
from pathlib import Path
from fastapi import FastAPI, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.openapi.utils import get_openapi
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from os import environ as env
from dotenv import find_dotenv, load_dotenv
from authlib.integrations.starlette_client import OAuth
from services.logger_singleton import LoggerSingleton
from memory.memory_graph import MemoryGraph

# Ensure SSL_CERT_FILE is valid before any httpx clients are created
# httpx reads SSL_CERT_FILE directly and fails if file doesn't exist
# This check happens early to prevent httpx errors when OpenAI client initializes
if 'SSL_CERT_FILE' in os.environ:
    ssl_cert_file = os.environ['SSL_CERT_FILE']
    # Handle empty string (httpx will fail if SSL_CERT_FILE is empty)
    if not ssl_cert_file or not ssl_cert_file.strip():
        del os.environ['SSL_CERT_FILE']
    else:
        cert_path = Path(ssl_cert_file)
        if not cert_path.exists() or not cert_path.is_file() or not os.access(ssl_cert_file, os.R_OK):
            # Unset if file doesn't exist or isn't readable - httpx will use system defaults
            del os.environ['SSL_CERT_FILE']
            try:
                logger = LoggerSingleton.get_logger(__name__)
                logger.warning(f"SSL_CERT_FILE points to non-existent or unreadable file: {ssl_cert_file}, unsetting to use system defaults")
            except:
                # Logger might not be initialized yet, just print
                print(f"WARNING: SSL_CERT_FILE points to non-existent or unreadable file: {ssl_cert_file}, unsetting to use system defaults")
from fastapi.security import HTTPBearer, APIKeyHeader
from core.services.telemetry import get_telemetry
from routers.v1 import v1_router
from routers.v1.user_routes import router as user_router
from routers.v1.memory_routes_v1 import router as memory_router
# document_routes_v2 is conditionally loaded in routers/v1/__init__.py (cloud-only, requires Temporal)
from routers.v1.feedback_routes import router as feedback_router
from azure.monitor.opentelemetry import configure_azure_monitor
from api_handlers.chat_gpt_completion import ChatGPTCompletion
from datastore.neo4jconnection import AsyncNeo4jConnection


# Load environment variables (conditionally based on USE_DOTENV)
use_dotenv = env.get("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    ENV_FILE = find_dotenv()
    if ENV_FILE:
        load_dotenv(ENV_FILE)

# Logger
logger = LoggerSingleton.get_logger(__name__)
logger.info("Logger initialized at top of app_factory.py!")

# Azure Monitor
if env.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    configure_azure_monitor()

api_key = env.get("OPENAI_API_KEY")
organization_id = env.get("OPENAI_ORGANIZATION")

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup_task = None
    mongodb_warmup_task = None
    app.state.httpx_client = None
    # Ensure AsyncNeo4jConnection singleton is reset before creating the app-scoped MemoryGraph
    # to avoid reusing asyncio primitives across different event loops in tests.
    try:
        AsyncNeo4jConnection._instance = None
    except Exception:
        pass
    
    # CRITICAL: Initialize MongoDB client BEFORE creating MemoryGraph
    # This ensures MemoryGraph.__init__ can access the shared MongoDB client
    from services.mongo_client import get_mongo_db
    import os
    logger.info("Pre-initializing MongoDB client singleton...")
    logger.info(f"DEBUG: MONGO_URI set: {bool(os.getenv('MONGO_URI'))}")
    logger.info(f"DEBUG: DATABASE_URI set: {bool(os.getenv('DATABASE_URI'))}")
    if os.getenv('DATABASE_URI'):
        logger.info(f"DEBUG: DATABASE_URI value: {os.getenv('DATABASE_URI')[:80]}...")
    shared_db = get_mongo_db()
    if shared_db is not None:
        logger.info(f"✅ MongoDB client initialized successfully: {shared_db.name}")
    else:
        logger.error("❌ MongoDB client initialization returned None - THIS WILL CAUSE FALLBACK TO PARSE SERVER!")
    
    app.state.memory_graph = MemoryGraph()

    # Initialize shared httpx client for connection pooling (auth/Parse/Qdrant-adjacent HTTP calls)
    try:
        app.state.httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=3.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=30.0),
            http2=True,
            follow_redirects=True,
        )
        logger.info("✅ Shared httpx client initialized for connection pooling")
    except Exception as e:
        logger.warning(f"Failed to initialize shared httpx client: {e}")
    
    # Initialize Qdrant with optimizations
    await app.state.memory_graph.init_qdrant()
    
    # Optimize existing collection if needed
    if app.state.memory_graph.qdrant_collection:
        try:
            await app.state.memory_graph.optimize_qdrant_collection()
        except Exception as e:
            logger.warning(f"Could not optimize Qdrant collection: {e}")

    # Warm up Qdrant connection to avoid first-search cold start
    try:
        await app.state.memory_graph.warm_qdrant_connection()
    except Exception as e:
        logger.warning(f"Qdrant warmup failed (non-critical): {e}")
    
    try:
        logger.info("Starting up application...")
        
        # Warm up MongoDB connection first (most critical for API performance)
        try:
            await app.state.memory_graph.warm_mongodb_connection()
            if app.state.memory_graph.mongodb_warmed:
                logger.info("Starting MongoDB keep-warm task...")
                mongodb_warmup_task = asyncio.create_task(
                    app.state.memory_graph.keep_mongodb_warm()
                )
                logger.info("MongoDB keep-warm task started")
        except Exception as e:
            logger.error(f"Error during MongoDB warmup: {e}")
            logger.warning("Application will continue running with slower first MongoDB queries")
        
        # Then warm up Neo4j connection
        try:
            # Ensure AsyncNeo4jConnection does not reuse a previous event loop's instance
            AsyncNeo4jConnection._instance = None
            logger.info("About to ensure async connection...")
            await app.state.memory_graph.ensure_async_connection()
            logger.info("Async connection ensured")
            if app.state.memory_graph.async_neo_conn.fallback_mode:
                logger.warning("Neo4j connection in fallback mode - some functionality will be limited")
            else:
                logger.info("Starting connection warm-up task...")
                await app.state.memory_graph.async_neo_conn.warm_connection()
                warmup_task = asyncio.create_task(
                    app.state.memory_graph.async_neo_conn.keep_warm()
                )
                logger.info("Warm-up task started")
            logger.info("Lifespan startup complete")
        except Exception as e:
            logger.error(f"Error during Neo4j initialization: {e}")
            logger.warning("Application will continue running with limited Neo4j functionality")
        
        # Yield control back to FastAPI
        yield
        
        logger.info("Lifespan yield complete")
        
    except GeneratorExit:
        # Handle graceful shutdown when generator is closed
        logger.info("GeneratorExit received - shutting down gracefully")
    except Exception as e:
        logger.error(f"Unexpected error in lifespan: {e}")
    finally:
        logger.info("Application shutting down")
        
        # Cancel MongoDB keep-warm task
        if mongodb_warmup_task and not mongodb_warmup_task.done():
            try:
                mongodb_warmup_task.cancel()
                await asyncio.wait_for(mongodb_warmup_task, timeout=5.0)
                logger.info("MongoDB keep-warm task cancelled successfully")
            except asyncio.TimeoutError:
                logger.warning("MongoDB keep-warm task cancellation timed out")
            except asyncio.CancelledError:
                logger.info("MongoDB keep-warm task was already cancelled")
            except Exception as e:
                logger.error(f"Error cancelling MongoDB keep-warm task: {e}")
        
        # Cancel Neo4j warm-up task
        if warmup_task and not warmup_task.done():
            try:
                warmup_task.cancel()
                # Use a timeout to prevent hanging
                await asyncio.wait_for(warmup_task, timeout=5.0)
                logger.info("Neo4j warm-up task cancelled successfully")
            except asyncio.TimeoutError:
                logger.warning("Neo4j warm-up task cancellation timed out")
            except asyncio.CancelledError:
                logger.info("Neo4j warm-up task was already cancelled")
            except Exception as e:
                logger.error(f"Error cancelling Neo4j warm-up task: {e}")
        
        # DON'T close MongoDB connection if it's the shared singleton
        # The singleton should persist across lifespan cycles for better performance
        # Only close if it's a non-shared client
        if hasattr(app.state.memory_graph, 'mongo_client') and app.state.memory_graph.mongo_client:
            try:
                # Check if this is the shared singleton client
                from services.mongo_client import get_mongo_db
                shared_db = get_mongo_db()
                is_shared_client = (shared_db is not None and
                                  app.state.memory_graph.mongo_client == shared_db.client)
                
                if not is_shared_client:
                    # Only close if it's a client we created ourselves
                    app.state.memory_graph.mongo_client.close()
                    logger.info("MongoDB connection closed successfully (non-shared client)")
                else:
                    logger.debug("Skipping MongoDB client close (shared singleton - managed by services.mongo_client)")
            except Exception as e:
                logger.warning(f"Error checking/closing MongoDB connection: {e}")
        
        # Close Neo4j connection if it exists
        if hasattr(app.state.memory_graph, 'async_neo_conn') and app.state.memory_graph.async_neo_conn:
            try:
                await app.state.memory_graph.async_neo_conn.close()
                logger.info("Neo4j connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing Neo4j connection: {e}")
        
        # Call MemoryGraph cleanup to close Qdrant and other connections
        if hasattr(app.state, 'memory_graph') and app.state.memory_graph:
            try:
                await app.state.memory_graph.cleanup()
                logger.info("MemoryGraph cleanup completed successfully")
            except Exception as e:
                logger.error(f"Error during MemoryGraph cleanup: {e}")

        # Close shared httpx client
        if getattr(app.state, "httpx_client", None):
            try:
                await app.state.httpx_client.aclose()
                logger.info("Shared httpx client closed successfully")
            except Exception as e:
                logger.warning(f"Error closing shared httpx client: {e}")

        # Telemetry cleanup (if needed in future)
        try:
            # Telemetry service handles its own cleanup
            telemetry = get_telemetry()
            logger.info("Telemetry service shutdown")
        except Exception as e:
            logger.warning(f"Telemetry shutdown warning: {e}")
        
        logger.info("Lifespan shutdown complete")

# Main app factory
def create_app() -> FastAPI:
    app = FastAPI(
        title="Papr Memory API",
        description=(
            "API for managing personal memory items with authentication and user-specific data.\n"
            "## Authentication\n"
            "This API supports three authentication methods:\n"
            "- **API Key**: Include your API key in the `X-API-Key` header\n"
            "  ```\n  X-API-Key: <your-api-key>\n  ```\n"
            "- **Session Token**: Include your session token in the `X-Session-Token` header\n"
            "  ```\n  X-Session-Token: <your-session-token>\n  ```\n"
            "- **Bearer Token**: Include your OAuth2 token from Auth0 in the `Authorization` header\n"
            "  ```\n  Authorization: Bearer <token>\n  ```\n"
            "All endpoints require one of these authentication methods."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        servers=[
            {"url": "https://memory.papr.ai", "description": "Production server"},
        ]
    )


    # Middleware
    app.add_middleware(
        SessionMiddleware, 
        secret_key=env.get("APP_SECRET_KEY", secrets.token_urlsafe(32)),
        max_age=3600,  # 1 hour
        same_site="lax",  # Less restrictive than 'strict', allows redirects
        https_only=False,  # Set to True in production with HTTPS
        session_cookie="papr_session"
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[f"https://{env.get('AUTH0_DOMAIN')}", env.get("PARSE_SERVER_URL"), env.get("WEB_APP_URL"), env.get("PYTHON_SERVER_URL")],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly allow OPTIONS for GraphiQL preflight
        allow_headers=["*"],  # Allow all headers including X-API-Key
        expose_headers=["*"]  # Expose all response headers
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.middleware("http")
    async def add_server_timing_headers(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Server-Processing-Ms"] = f"{duration_ms:.2f}"
        existing_server_timing = response.headers.get("Server-Timing")
        timing_value = f"app;dur={duration_ms:.2f}"
        if existing_server_timing:
            response.headers["Server-Timing"] = f"{existing_server_timing}, {timing_value}"
        else:
            response.headers["Server-Timing"] = timing_value
        return response
    
    # Add authentication middleware to support request.auth
    from starlette.middleware.authentication import AuthenticationMiddleware
    from starlette.authentication import AuthenticationBackend, AuthCredentials, SimpleUser
    
    class DummyAuthBackend(AuthenticationBackend):
        async def authenticate(self, request):
            # This is a dummy backend that doesn't actually authenticate
            # but allows request.auth to be accessed without errors
            return None
    
    app.add_middleware(AuthenticationMiddleware, backend=DummyAuthBackend())

    # Static files - Commented out to allow dynamic ai-plugin.json
    # app.mount("/.well-known", StaticFiles(directory=".well-known"), name="well-known")

    # Security schemes
    bearer_auth = HTTPBearer(scheme_name="Bearer", auto_error=False)
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
    session_token_header = APIKeyHeader(name="X-Session-Token", auto_error=False)
    app.state.bearer_auth = bearer_auth
    app.state.api_key_header = api_key_header
    app.state.session_token_header = session_token_header

    # Routers
    # Routers are now included in routers/v1/__init__.py
    # v1_router.include_router(user_router)
    # v1_router.include_router(memory_router)
    # v1_router.include_router(feedback_router)
    #v1_router.include_router(document_router)
    app.include_router(v1_router)

    # JWKS endpoint at root level (/.well-known/jwks.json)
    # Only load in cloud edition (requires Auth0)
    from config.features import get_features
    features = get_features()
    if features.has_auth0:
        try:
            from routers.v1.jwks_routes import router as jwks_router
            app.include_router(jwks_router)
            logger.info("JWKS routes loaded - Auth0 enabled")
        except ImportError as e:
            logger.warning(f"JWKS routes not available: {e}")
    else:
        logger.info("JWKS routes disabled - Auth0 not enabled (cloud-only feature)")

    # Initialize telemetry service (lazy initialization)
    # Telemetry is automatically initialized when first used via get_telemetry()
    # No need to store in app.state anymore
    logger.info("Telemetry service ready (will initialize on first use)")

    # OAuth setup (moved inside create_app)
    oauth = OAuth()
    oauth.register(
        name="auth0_browser_extension",
        client_id=env.get("AUTH0_CLIENT_ID_BROWSER"),
        client_secret=env.get("AUTH0_CLIENT_SECRET_BROWSER"),
        client_kwargs={"scope": "openid profile email offline_access"},
        server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration'
    )
    oauth.register(
        name="auth0_papr_plugin",
        client_id=env.get("AUTH0_CLIENT_ID_PAPR"),
        client_secret=env.get("AUTH0_CLIENT_SECRET_PAPR"),
        client_kwargs={"scope": "openid profile email offline_access"},
        server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration'
    )
    app.state.oauth = oauth

    # Final check: Ensure SSL_CERT_FILE is valid before initializing OpenAI client
    # httpx (used by OpenAI) reads SSL_CERT_FILE directly and fails if file doesn't exist
    if 'SSL_CERT_FILE' in os.environ:
        ssl_cert_file = os.environ['SSL_CERT_FILE']
        # Handle empty string (httpx will fail if SSL_CERT_FILE is empty)
        if not ssl_cert_file or not ssl_cert_file.strip():
            del os.environ['SSL_CERT_FILE']
            logger.debug("SSL_CERT_FILE was empty, unset to use system defaults")
        else:
            cert_path = Path(ssl_cert_file)
            if not cert_path.exists() or not cert_path.is_file() or not os.access(ssl_cert_file, os.R_OK):
                # Unset if file doesn't exist or isn't readable - httpx will use system defaults
                del os.environ['SSL_CERT_FILE']
                logger.warning(f"SSL_CERT_FILE points to non-existent or unreadable file: {ssl_cert_file}, unsetting to use system defaults")
            else:
                logger.debug(f"SSL_CERT_FILE validated: {ssl_cert_file}")

    chat_gpt = ChatGPTCompletion(api_key, organization_id, env.get("LLM_MODEL"), env.get("LLM_LOCATION_CLOUD", default=True), env.get("EMBEDDING_MODEL_LOCAL"))
    app.state.chat_gpt = chat_gpt

    # Optional local/test webhook receiver so Temporal activities can POST to http://test/webhook-test
    try:
        import os as _os
        from fastapi import Request as _Request
        if _os.getenv("PAPR_ENABLE_TEST_WEBHOOK", "").lower() in {"1", "true", "yes"}:
            async def _test_webhook_receiver(request: _Request):
                try:
                    payload = await request.json()
                except Exception:
                    payload = None
                return {"status": "ok", "received": bool(payload)}
            app.add_api_route("/webhook-test", _test_webhook_receiver, methods=["POST"])  # type: ignore[arg-type]
    except Exception:
        pass

    # Custom OpenAPI as closure so it can access 'app'
    def custom_v1_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        from fastapi.openapi.utils import get_openapi
        import json, yaml
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,            
            description=app.description,
            routes=app.routes,
        )
        
        # Ensure servers field is preserved/added
        server_url = env.get("PYTHON_SERVER_URL", "https://memory.papr.ai")
        openapi_schema["servers"] = [
            {"url": server_url, "description": "Production server"},
        ]
        
        # Patch security scheme names and formats for better OpenAPI docs
        existing_security_schemes = openapi_schema.get("components", {}).get("securitySchemes", {})
        logger.info(f"Auto-detected security schemes before patch: {existing_security_schemes}")

        # Patch Bearer to have bearerFormat: JWT and a good description
        if "Bearer" in existing_security_schemes:
            existing_security_schemes["Bearer"]["bearerFormat"] = "JWT"
            existing_security_schemes["Bearer"]["description"] = "Bearer token (JWT) from Auth0 OAuth2 or similar"

        # Patch APIKeyHeader to X-API-Key if present
        if "APIKeyHeader" in existing_security_schemes and existing_security_schemes["APIKeyHeader"].get("name") == "X-API-Key":
            existing_security_schemes["X-API-Key"] = existing_security_schemes.pop("APIKeyHeader")
            existing_security_schemes["X-API-Key"]["description"] = "API key for authentication"

        # Patch APIKeyHeader to X-Session-Token if present
        if "APIKeyHeader" in existing_security_schemes and existing_security_schemes["APIKeyHeader"].get("name") == "X-Session-Token":
            existing_security_schemes["X-Session-Token"] = existing_security_schemes.pop("APIKeyHeader")
            existing_security_schemes["X-Session-Token"]["description"] = "Session token for authentication"

        # Force-add X-API-Key if missing
        if "X-API-Key" not in existing_security_schemes:
            existing_security_schemes["X-API-Key"] = {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for authentication"
            }

        # Patch all security arrays in all paths/operations to rename APIKeyHeader to X-API-Key
        for path_item in openapi_schema.get("paths", {}).values():
            for operation in path_item.values():
                if isinstance(operation, dict) and "security" in operation:
                    for sec in operation["security"]:
                        if "APIKeyHeader" in sec:
                            sec["X-API-Key"] = sec.pop("APIKeyHeader")

        # Ensure OAuth2 is present for ChatGPT plugin compatibility
        if "OAuth2" not in existing_security_schemes:
            server_url = env.get("PYTHON_SERVER_URL", "https://memory.papr.ai")
            existing_security_schemes["OAuth2"] = {
                "type": "oauth2",
                "flows": {
                    "authorizationCode": {
                        "authorizationUrl": f"{server_url}/login",
                        "tokenUrl": f"{server_url}/token",
                        "refreshUrl": f"{server_url}/token",
                        "scopes": {
                            "openid": "OpenID",
                            "profile": "Profile access",
                            "email": "Email access",
                            "offline_access": "Offline access"
                        }
                    }
                }
            }
        openapi_schema["components"]["securitySchemes"] = existing_security_schemes
        logger.info(f"Auto-detected security schemes after patch: {existing_security_schemes}")
        
        # Filter to only include v1 paths and auth endpoints
        filtered_paths = {}
        for path, data in openapi_schema["paths"].items():
            if path.startswith("/v1/") or path in ["/login", "/logout", "/token", "/me", "/callback"]:
                filtered_paths[path] = data
        openapi_schema["paths"] = filtered_paths
        
        with open("openapi.json", "w") as f:
            json.dump(openapi_schema, f, indent=2)
        with open("openapi.yaml", "w") as f:
            yaml.dump(openapi_schema, f, sort_keys=False)
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    app.openapi = custom_v1_openapi



    return app
