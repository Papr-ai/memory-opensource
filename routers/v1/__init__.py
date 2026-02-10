from fastapi import APIRouter
import logging

from config.features import get_features

logger = logging.getLogger(__name__)

v1_router = APIRouter(prefix="/v1", tags=["v1"])

# Get feature flags to determine which routes to load
features = get_features()

# ==========================================
# Core routes (always loaded - open source)
# ==========================================
from .memory_routes_v1 import router as memory_router
from .user_routes import router as user_router
from .feedback_routes import router as feedback_router
from .websocket_routes import router as websocket_router
from .schema_routes_v1 import router as schema_router
from .message_routes import router as message_router
from .omo_routes import router as omo_router

# Include core routers (always available)
v1_router.include_router(memory_router)
v1_router.include_router(user_router)
v1_router.include_router(feedback_router)
v1_router.include_router(websocket_router)
v1_router.include_router(schema_router)
v1_router.include_router(message_router)
v1_router.include_router(omo_router)

# ==========================================
# Cloud-only routes (conditionally loaded)
# ==========================================

# Sync routes (cloud-only feature)
if features.is_cloud:
    try:
        from .sync_routes import router as sync_router
        v1_router.include_router(sync_router)
        logger.info("Sync routes loaded - cloud edition")
    except ImportError as e:
        logger.warning(f"Sync routes not available: {e}")
else:
    logger.info("Sync routes disabled - requires cloud edition")

# Telemetry routes (cloud-only - for sending telemetry to cloud)
if features.is_cloud:
    try:
        from .telemetry_routes import router as telemetry_router
        v1_router.include_router(telemetry_router)
        logger.info("Telemetry routes loaded - cloud edition")
    except ImportError as e:
        logger.warning(f"Telemetry routes not available: {e}")
else:
    logger.info("Telemetry routes disabled - requires cloud edition")

# Document routes (requires Temporal for durable processing)
if features.is_enabled("temporal"):
    try:
        from .document_routes_v2 import router as document_v2_router
        v1_router.include_router(document_v2_router)
        logger.info("Document routes (v2) loaded - Temporal enabled")
    except ImportError as e:
        logger.warning(f"Document routes not available: {e}")
else:
    logger.info("Document routes (v2) disabled - Temporal not enabled (cloud-only feature)")

# GraphQL routes (requires Neo4j Aura GraphQL endpoint)
if features.is_cloud:
    try:
        from .graphql_routes import router as graphql_router
        v1_router.include_router(graphql_router)
        logger.info("GraphQL routes loaded - cloud edition")
    except ImportError as e:
        logger.warning(f"GraphQL routes not available: {e}")
else:
    logger.info("GraphQL routes disabled - requires cloud edition (Neo4j Aura)")

# JWKS router should be registered at root level (not under /v1)
# We'll register it separately in app_factory.py (cloud-only, requires Auth0) 