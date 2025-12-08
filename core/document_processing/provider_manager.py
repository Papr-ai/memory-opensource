"""
Document provider management and factory classes
"""

import os
# Load environment variables in this module for provider configs
try:
    from dotenv import find_dotenv, load_dotenv
    # Respect USE_DOTENV setting
    use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
    if use_dotenv:
        _ENV_FILE = find_dotenv()
        if _ENV_FILE:
            load_dotenv(_ENV_FILE)
except Exception:
    pass
from typing import Dict, Any, Optional, List
from .providers.base import DocumentProvider
from .providers.tensorlake import TensorLakeProvider
from .providers.reducto import ReductoProvider
from .providers.gemini import GeminiVisionProvider
from .providers.deepseek import DeepSeekOCRProvider
from .providers.paddleocr import PaddleOCRProvider
from services.logger_singleton import LoggerSingleton
from pydantic import BaseModel

logger = LoggerSingleton.get_logger(__name__)


class ProviderConfig(BaseModel):
    provider_name: str
    config: Dict[str, Any]
    priority: int = 1
    enabled: bool = True


class TenantDocumentConfig(BaseModel):
    organization_id: str
    namespace: Optional[str] = None
    providers: List[ProviderConfig]
    fallback_strategy: str = "next_priority"  # next_priority, fail, default
    default_provider: Optional[str] = None
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_formats: List[str] = ["pdf", "png", "jpg", "jpeg", "webp", "html", "txt"]
    webhook_url: Optional[str] = None


class ProviderRegistry:
    """Registry for document processing providers"""

    _providers = {
        "tensorlake": TensorLakeProvider,
        "reducto": ReductoProvider,
        "gemini": GeminiVisionProvider,
        "deepseek-ocr": DeepSeekOCRProvider,
        "paddleocr": PaddleOCRProvider
    }

    @classmethod
    def get_provider_class(cls, provider_name: str) -> type:
        return cls._providers.get(provider_name.lower())

    @classmethod
    def list_providers(cls) -> List[str]:
        return list(cls._providers.keys())

    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Register a new provider class"""
        cls._providers[name.lower()] = provider_class


class TenantConfigManager:
    """Manages document processing configuration per tenant"""

    def __init__(self, memory_graph=None):
        self.memory_graph = memory_graph

    async def get_tenant_config(
        self,
        organization_id: str,
        namespace: Optional[str] = None
    ) -> TenantDocumentConfig:
        """Get document processing config for a tenant"""

        if self.memory_graph:
            try:
                # Query from Neo4j
                await self.memory_graph.ensure_async_connection()
                async with self.memory_graph.async_neo_conn.get_session() as session:
                    query = """
                    MATCH (org:Organization {id: $org_id})
                    OPTIONAL MATCH (org)-[:HAS_CONFIG]->(config:DocumentConfig)
                    WHERE config.namespace = $namespace OR ($namespace IS NULL AND config.namespace IS NULL)
                    RETURN config
                    """

                    result = await session.run(query, org_id=organization_id, namespace=namespace)
                    record = await result.single()

                    if record and record["config"]:
                        config_data = dict(record["config"])
                        return TenantDocumentConfig.model_validate(config_data)
            except Exception as e:
                logger.error(f"Error fetching tenant config from Neo4j: {e}")

        # Return default configuration with Reducto as primary and TensorLake as fallback
        providers = []

        # Reducto as primary provider (best for structured extraction)
        reducto_api_key = os.getenv("REDUCTO_API_KEY")
        if reducto_api_key:
            # Normalize environment; Reducto SDK wants 'production', 'eu', 'au' (not 'us')
            env = (os.getenv("REDUCTO_ENVIRONMENT", "us") or "us").lower()
            if env == "us":
                env = "production"
            elif env not in {"production", "eu", "au"}:
                logger.warning(f"Unknown REDUCTO_ENVIRONMENT: {env}; defaulting to 'production'")
                env = "production"

            providers.append(ProviderConfig(
                provider_name="reducto",
                config={
                    "api_key": reducto_api_key,
                    "environment": env,
                    "timeout": 120
                },
                priority=1  # Primary provider
            ))

        # TensorLake as secondary provider
        tensorlake_api_key = os.getenv("TENSORLAKE_API_KEY")
        if tensorlake_api_key:
            providers.append(ProviderConfig(
                provider_name="tensorlake",
                config={
                    "api_key": tensorlake_api_key,
                    "base_url": os.getenv("TENSORLAKE_BASE_URL", "https://api.tensorlake.ai"),
                    "timeout": 120
                },
                priority=2  # Secondary/fallback provider
            ))

        # Gemini as fallback provider
        gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            providers.append(ProviderConfig(
                provider_name="gemini",
                config={
                    "api_key": gemini_api_key,
                    "model": "gemini-2.5-flash"
                },
                priority=3
            ))

        # If no providers available, add a minimal config for testing
        if not providers:
            logger.warning("No document processing provider API keys found in environment")
            providers.append(ProviderConfig(
                provider_name="gemini",
                config={
                    "api_key": "test_key",  # Will fail validation but allows testing
                    "model": "gemini-2.5-flash"
                },
                priority=1,
                enabled=False
            ))

        return TenantDocumentConfig(
            organization_id=organization_id,
            namespace=namespace,
            providers=providers
        )

    async def save_tenant_config(self, config: TenantDocumentConfig) -> bool:
        """Save document processing config for a tenant"""

        if not self.memory_graph:
            logger.warning("No memory graph available for saving tenant config")
            return False

        try:
            await self.memory_graph.ensure_async_connection()
            async with self.memory_graph.async_neo_conn.get_session() as session:
                query = """
                MATCH (org:Organization {id: $org_id})
                MERGE (org)-[:HAS_CONFIG]->(config:DocumentConfig {namespace: $namespace})
                SET config += $config_data
                RETURN config
                """

                config_data = config.model_dump()
                result = await session.run(
                    query,
                    org_id=config.organization_id,
                    namespace=config.namespace,
                    config_data=config_data
                )
                return await result.single() is not None
        except Exception as e:
            logger.error(f"Error saving tenant config to Neo4j: {e}")
            return False


class DocumentProcessorFactory:
    """Factory for creating document processors with tenant-specific configuration"""

    def __init__(self, config_manager: TenantConfigManager):
        self.config_manager = config_manager
        self.registry = ProviderRegistry()

    async def create_processor(
        self,
        organization_id: str,
        namespace: Optional[str] = None,
        preferred_provider: Optional[str] = None
    ) -> DocumentProvider:
        """Create a document processor for a tenant"""

        tenant_config = await self.config_manager.get_tenant_config(organization_id, namespace)

        # Sort providers by priority
        sorted_providers = sorted(tenant_config.providers, key=lambda p: p.priority)

        # Try preferred provider first if specified
        if preferred_provider:
            for provider_config in sorted_providers:
                if provider_config.provider_name == preferred_provider and provider_config.enabled:
                    try:
                        provider = await self._create_provider_instance(provider_config)
                        if await provider.health_check():
                            logger.info(f"Using preferred provider: {preferred_provider}")
                            return provider
                    except Exception as e:
                        # Preferred provider not available; continue to fallback providers
                        logger.warning(f"Preferred provider {preferred_provider} unavailable: {e}. Falling back to next provider.")

        # Try providers in priority order; capture first healthy fallback if preferred fails
        last_error = None
        healthy_fallback = None
        for provider_config in sorted_providers:
            if not provider_config.enabled:
                continue
            try:
                provider = await self._create_provider_instance(provider_config)
                if await provider.health_check():
                    logger.info(f"Provider healthy: {provider_config.provider_name}")
                    if not healthy_fallback:
                        healthy_fallback = provider
                    # Prefer TensorLake only if explicitly requested; otherwise favor Reducto when available
                    if provider_config.provider_name == "reducto" and (preferred_provider in (None, "reducto")):
                        logger.info("Selecting Reducto as primary for structured extraction")
                        return provider
            except Exception as e:
                last_error = e
                logger.error(f"Provider {provider_config.provider_name} failed: {e}")
                continue

        if healthy_fallback:
            logger.info("Falling back to first healthy provider")
            return healthy_fallback

        raise ValueError(f"No healthy document processing provider available{f' (last error: {last_error})' if last_error else ''}")

    async def _create_provider_instance(self, provider_config: ProviderConfig) -> DocumentProvider:
        """Create a provider instance from configuration"""

        provider_class = self.registry.get_provider_class(provider_config.provider_name)
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider_config.provider_name}")

        provider = provider_class(provider_config.config)

        if not await provider.validate_config():
            raise ValueError(f"Invalid configuration for provider: {provider_config.provider_name}")

        return provider