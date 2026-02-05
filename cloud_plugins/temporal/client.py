"""
Temporal Client Wrapper

Manages connection to Temporal server for durable workflow execution.
"""

import os
from dotenv import load_dotenv, find_dotenv
from typing import Optional
from services.logger_singleton import LoggerSingleton

try:
    from temporalio.client import Client, TLSConfig
    from temporalio.converter import DataConverter
    from temporalio.contrib.pydantic import pydantic_data_converter
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    Client = None
    TLSConfig = None

logger = LoggerSingleton.get_logger(__name__)

# Ensure .env variables are loaded for both API and worker processes
_ENV_FILE = find_dotenv()
if _ENV_FILE:
    load_dotenv(_ENV_FILE)

_temporal_client: Optional[Client] = None


async def get_temporal_client() -> Client:
    """
    Get or create Temporal client.

    Returns:
        Temporal client instance

    Raises:
        ConnectionError: If cannot connect to Temporal server
    """
    global _temporal_client

    if not TEMPORAL_AVAILABLE:
        raise ConnectionError("Temporal library (temporalio) is not installed")

    if _temporal_client is None:
        temporal_address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
        temporal_namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
        temporal_api_key = os.getenv("TEMPORAL_API_KEY")

        # Auto-detect when to enable TLS (Temporal Cloud) or use env override
        addr_lower = temporal_address.lower()
        use_tls_env = os.getenv("TEMPORAL_USE_TLS", "").strip().lower() in {"1", "true", "yes", "on"}
        use_tls = use_tls_env or any(s in addr_lower for s in ["tmprl.cloud", "api.temporal.io", ".aws.api.temporal.io"])

        # Optional custom root CA (rarely needed for Temporal Cloud)
        server_root_ca_cert = None
        ca_path = os.getenv("TEMPORAL_SERVER_ROOT_CA", "").strip()
        if ca_path:
            try:
                with open(ca_path, "rb") as f:
                    server_root_ca_cert = f.read()
            except Exception as e:
                logger.warning(f"Could not read TEMPORAL_SERVER_ROOT_CA at '{ca_path}': {e}")

        tls_config = TLSConfig(server_root_ca_cert=server_root_ca_cert) if use_tls else None

        try:
            logger.info(
                f"Connecting to Temporal at {temporal_address}, namespace: {temporal_namespace}"
            )

            # Build connection arguments - use api_key parameter directly
            connect_kwargs = {
                "namespace": temporal_namespace,
            }

            if tls_config is not None:
                connect_kwargs["tls"] = tls_config

            # Pass API key directly for Temporal Cloud authentication
            if temporal_api_key:
                connect_kwargs["api_key"] = temporal_api_key

            _temporal_client = await Client.connect(
                temporal_address,
                data_converter=pydantic_data_converter,
                **connect_kwargs,
            )

            logger.info("Successfully connected to Temporal")

        except Exception as e:
            logger.error(f"Failed to connect to Temporal: {e}")
            raise ConnectionError(f"Cannot connect to Temporal server: {e}")
    
    return _temporal_client


async def check_temporal_health() -> bool:
    """
    Check if Temporal server is healthy.

    Returns:
        True if healthy, False otherwise
    """
    if not TEMPORAL_AVAILABLE:
        logger.warning("Temporal library not available")
        return False

    try:
        client = await get_temporal_client()
        # Try to list workflows to verify connection
        await client.list_workflows().list()
        return True
    except Exception as e:
        logger.warning(f"Temporal health check failed: {e}")
        return False

