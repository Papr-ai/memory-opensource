"""
Feature Flag System for Papr Memory

Simple config-based feature flags to manage open source vs cloud features.
No external service needed - just YAML files and environment variables.

Usage:
    from config import get_features
    
    features = get_features()
    
    if features.has_stripe:
        # Load Stripe integration
        pass
    
    if features.is_cloud:
        # Cloud-specific logic
        pass
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class FeatureFlags:
    """
    Feature flag system for managing open source vs cloud features.
    
    Features are loaded from YAML config files based on PAPR_EDITION env var.
    - opensource edition: Loads config/opensource.yaml
    - cloud edition: Loads config/cloud.yaml
    
    This allows the same codebase to conditionally load features.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern - one instance per process"""
        if cls._instance is None:
            cls._instance = super(FeatureFlags, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize feature flags from config files"""
        if self._initialized:
            return

        # Load environment variables from .env file (if enabled)
        # Respect USE_DOTENV setting for consistency with main.py
        use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
        if use_dotenv:
            load_dotenv()

        self.edition = os.getenv("PAPR_EDITION", "opensource").lower()
        self.config_dir = Path(__file__).parent
        
        # Load base config (shared settings)
        self.base_config = self._load_config("base.yaml")
        
        # Load edition-specific config
        if self.edition == "cloud":
            self.edition_config = self._load_config("cloud.yaml")
            # If cloud.yaml doesn't exist (e.g., in OSS repo), fallback to opensource.yaml
            if not self.edition_config:
                logger.warning("cloud.yaml not found - cloud features disabled. Using opensource.yaml as fallback.")
                self.edition_config = self._load_config("opensource.yaml")
        else:
            self.edition_config = self._load_config("opensource.yaml")
        
        # Merge configs (edition-specific overrides base)
        self.config = {**self.base_config, **self.edition_config}
        
        self._initialized = True
        
        logger.info(f"FeatureFlags initialized for edition: {self.edition}")
        logger.debug(f"Loaded features: {self.enabled_features}")
    
    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load YAML config file"""
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config or {}
        except Exception as e:
            logger.error(f"Error loading config {filename}: {e}")
            return {}
    
    # ==========================================
    # Edition Detection
    # ==========================================
    
    @property
    def is_cloud(self) -> bool:
        """Check if running cloud edition"""
        return self.edition == "cloud"
    
    @property
    def is_opensource(self) -> bool:
        """Check if running open source edition"""
        return self.edition == "opensource"
    
    # ==========================================
    # Feature Checks
    # ==========================================
    
    def is_enabled(self, feature: str) -> bool:
        """
        Check if a feature is enabled.
        
        Args:
            feature: Feature name (e.g., "stripe", "auth0")
        
        Returns:
            True if feature is enabled, False otherwise
        """
        features = self.config.get("features", {})
        return features.get(feature, False)
    
    @property
    def enabled_features(self) -> list[str]:
        """Get list of all enabled features"""
        features = self.config.get("features", {})
        return [name for name, enabled in features.items() if enabled]
    
    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """
        Check if a plugin is enabled.
        
        Args:
            plugin_name: Plugin name (e.g., "stripe", "posthog")
        
        Returns:
            True if plugin is enabled, False otherwise
        """
        plugins = self.config.get("plugins", {})
        enabled_plugins = plugins.get("enabled", [])
        return plugin_name in enabled_plugins
    
    # ==========================================
    # Cloud Service Features
    # ==========================================
    
    @property
    def has_stripe(self) -> bool:
        """Check if Stripe payment processing is available"""
        return self.is_enabled("stripe") and os.getenv("STRIPE_SECRET_KEY")
    
    @property
    def has_auth0(self) -> bool:
        """Check if Auth0 OAuth is available"""
        return self.is_enabled("auth0") and os.getenv("AUTH0_DOMAIN")
    
    @property
    def has_amplitude(self) -> bool:
        """Check if Amplitude analytics is available"""
        return self.is_enabled("amplitude") and os.getenv("AMPLITUDE_API_KEY")
    
    @property
    def has_azure(self) -> bool:
        """Check if Azure services are available"""
        return self.is_enabled("azure_service_bus") and os.getenv("AZURE_SERVICE_BUS_CONNECTION_STRING")
    
    # ==========================================
    # Open Source Features
    # ==========================================
    
    @property
    def has_posthog(self) -> bool:
        """Check if PostHog analytics is available"""
        return self.is_enabled("posthog")
    
    @property
    def telemetry_enabled(self) -> bool:
        """
        Check if telemetry is enabled (respects opt-out).
        
        Users can disable via TELEMETRY_ENABLED=false env var.
        """
        if os.getenv("TELEMETRY_ENABLED", "true").lower() == "false":
            return False
        return self.is_enabled("telemetry")
    
    @property
    def telemetry_provider(self) -> str:
        """
        Get the telemetry provider to use.
        
        Returns:
            'posthog' for open source (self-hostable)
            'amplitude' for cloud
            'none' if telemetry disabled
        """
        if not self.telemetry_enabled:
            return "none"
        
        # Check explicit env var first
        provider = os.getenv("TELEMETRY_PROVIDER", "").lower()
        if provider in ["posthog", "amplitude", "none"]:
            return provider
        
        # Default based on edition
        if self.is_cloud and self.has_amplitude:
            return "amplitude"
        elif self.has_posthog:
            return "posthog"
        
        return "none"
    
    # ==========================================
    # Subscription & Limits
    # ==========================================
    
    @property
    def requires_subscription(self) -> bool:
        """Check if subscriptions are required (cloud only)"""
        subscription_config = self.config.get("subscription", {})
        return subscription_config.get("require_payment", False)
    
    def get_tier_limits(self, tier: str) -> Optional[Dict[str, Any]]:
        """
        Get limits for a subscription tier.
        
        Args:
            tier: Tier name (e.g., "pro", "starter")
        
        Returns:
            Dict with limits, or None if no limits (unlimited)
        """
        limits = self.config.get("limits", {})
        
        # In open source, no limits
        if self.is_opensource:
            return None
        
        return limits.get(tier, {})
    
    # ==========================================
    # Configuration Access
    # ==========================================
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value by key.
        
        Args:
            key: Config key (supports dot notation, e.g., "app.name")
            default: Default value if key not found
        
        Returns:
            Config value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    # ==========================================
    # Batch Processing Configuration
    # ==========================================
    
    def get_max_batch_size(self) -> int:
        """Get maximum batch size for current edition"""
        return self.config.get("batch_processing", {}).get("max_batch_size", 50)
    
    def get_temporal_threshold(self) -> int:
        """Get batch size threshold to trigger Temporal (cloud only)"""
        return self.config.get("temporal", {}).get("temporal_threshold", 100)
    
    def get_batch_limit_message(self) -> str:
        """Get message to show when batch limit exceeded"""
        return self.config.get("messaging", {}).get("batch_limit_exceeded", "")


# ==========================================
# Global Instance
# ==========================================

_features_instance: Optional[FeatureFlags] = None


def get_features() -> FeatureFlags:
    """
    Get the global FeatureFlags instance.
    
    Usage:
        from config import get_features
        
        features = get_features()
        if features.has_stripe:
            # Load Stripe
            pass
    
    Returns:
        Singleton FeatureFlags instance
    """
    global _features_instance
    
    if _features_instance is None:
        _features_instance = FeatureFlags()
    
    return _features_instance


# ==========================================
# Convenience Functions
# ==========================================

def is_cloud() -> bool:
    """Check if running cloud edition"""
    return get_features().is_cloud


def is_opensource() -> bool:
    """Check if running open source edition"""
    return get_features().is_opensource


def is_feature_enabled(feature: str) -> bool:
    """Check if a feature is enabled"""
    return get_features().is_enabled(feature)

