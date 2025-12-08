"""
Privacy-First Telemetry Service for Papr Memory

This service implements the principles from:
https://1984.vc/docs/founders-handbook/eng/open-source-telemetry/

Key Principles:
1. Privacy First - Anonymous by default, no PII ever collected
2. Easy Opt-Out - Multiple ways to disable (env var, flag, config)
3. Transparent - Users know exactly what's collected
4. Fail Silently - Never interrupts user experience
5. Self-Hostable - PostHog support for complete control

What We Collect:
- ✅ Feature usage (which endpoints, features used)
- ✅ Error types (anonymous error tracking)
- ✅ Performance metrics (response times, query speeds)

What We NEVER Collect:
- ❌ Memory content
- ❌ Search queries
- ❌ User personal data
- ❌ IP addresses
- ❌ File paths or names
- ❌ Unique device identifiers
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
import hashlib
import json

logger = logging.getLogger(__name__)


class TelemetryProvider(Enum):
    """Supported telemetry providers"""
    PAPR_PROXY = "papr_proxy"  # Default: Send to Papr's telemetry proxy (memory.papr.ai/v1/telemetry/events)
    POSTHOG = "posthog"        # Self-hostable, open source
    AMPLITUDE = "amplitude"    # Direct Amplitude (cloud-only)
    NONE = "none"              # Telemetry disabled


class TelemetryService:
    """
    Privacy-first telemetry service for Papr Memory.
    
    This service provides anonymous usage analytics while respecting user privacy.
    
    Default Behavior (OSS):
    - Sends anonymous telemetry to https://memory.papr.ai/v1/telemetry/events
    - The proxy endpoint forwards to Amplitude using Papr's secure API key
    - No API keys are exposed to clients
    
    Alternative Providers:
    - PostHog (self-hosted): Set TELEMETRY_PROVIDER=posthog
    - Amplitude (direct): Set TELEMETRY_PROVIDER=amplitude
    
    Usage:
        from core.services.telemetry import get_telemetry
        
        telemetry = get_telemetry()
        
        # Track a feature usage
        await telemetry.track("memory_created", {
            "type": "text",
            "has_metadata": True
        })
        
        # Track an error
        await telemetry.track_error("database_error", {
            "error_type": "connection_timeout"
        })
    """
    
    _instance: Optional['TelemetryService'] = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(TelemetryService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize telemetry service"""
        if self._initialized:
            return
        
        self.enabled = self._check_enabled()
        self.provider = self._detect_provider()
        self.anonymous_id = self._generate_anonymous_id()
        
        # Initialize the appropriate provider
        self._client = None
        if self.enabled and self.provider != TelemetryProvider.NONE:
            self._initialize_provider()
        
        self._initialized = True
        
        if self.enabled:
            logger.info(f"Telemetry enabled with provider: {self.provider.value}")
        else:
            logger.info("Telemetry disabled")
    
    def _check_enabled(self) -> bool:
        """
        Check if telemetry is enabled.
        
        Users can disable via TELEMETRY_ENABLED=false
        """
        enabled_str = os.getenv("TELEMETRY_ENABLED", "true").lower()
        return enabled_str in ["true", "1", "yes"]
    
    def _detect_provider(self) -> TelemetryProvider:
        """Detect which telemetry provider to use"""
        if not self.enabled:
            return TelemetryProvider.NONE
        
        # Default to Papr's telemetry proxy for anonymous OSS adoption tracking
        provider_str = os.getenv("TELEMETRY_PROVIDER", "papr_proxy").lower()
        
        try:
            return TelemetryProvider(provider_str)
        except ValueError:
            logger.warning(f"Unknown telemetry provider: {provider_str}, defaulting to papr_proxy")
            return TelemetryProvider.PAPR_PROXY
    
    def _generate_anonymous_id(self) -> str:
        """
        Generate an anonymous session ID.
        
        This ID is NOT persisted across restarts and is hashed from
        system info to ensure no unique device tracking.
        """
        # Use a combination that changes per session
        session_data = f"{os.getpid()}-{datetime.now().isoformat()}"
        return hashlib.sha256(session_data.encode()).hexdigest()[:16]
    
    def _initialize_provider(self):
        """Initialize the telemetry provider client"""
        try:
            if self.provider == TelemetryProvider.PAPR_PROXY:
                self._initialize_papr_proxy()
            elif self.provider == TelemetryProvider.POSTHOG:
                self._initialize_posthog()
            elif self.provider == TelemetryProvider.AMPLITUDE:
                self._initialize_amplitude()
        except Exception as e:
            logger.warning(f"Failed to initialize telemetry provider: {e}")
            self.enabled = False
            self.provider = TelemetryProvider.NONE
    
    def _initialize_papr_proxy(self):
        """
        Initialize Papr's telemetry proxy endpoint.
        
        This is the default provider for OSS installations. It sends anonymous
        telemetry to memory.papr.ai/v1/telemetry/events, which then forwards
        to Amplitude using Papr's secure API key (never exposed to clients).
        
        This allows OSS users to help track adoption without exposing API keys.
        """
        try:
            import httpx
            
            # Get the proxy URL (defaults to memory.papr.ai)
            proxy_url = os.getenv("PAPR_TELEMETRY_URL", "https://memory.papr.ai")
            # Ensure it doesn't end with a slash
            proxy_url = proxy_url.rstrip('/')
            
            # Full endpoint URL
            self._proxy_url = f"{proxy_url}/v1/telemetry/events"
            
            # Create a persistent HTTP client for efficiency
            self._client = httpx.AsyncClient(timeout=5.0)
            
            logger.info(f"Papr telemetry proxy initialized: {self._proxy_url}")
            
        except ImportError:
            logger.warning("httpx not available for telemetry proxy. Install with: pip install httpx")
            self.enabled = False
    
    def _initialize_posthog(self):
        """Initialize PostHog (self-hostable)"""
        try:
            import posthog
            
            api_key = os.getenv("POSTHOG_API_KEY")
            host = os.getenv("POSTHOG_HOST", "https://app.posthog.com")
            
            if not api_key:
                logger.info("PostHog API key not configured, telemetry disabled")
                self.enabled = False
                return
            
            posthog.api_key = api_key
            posthog.host = host
            
            self._client = posthog
            logger.info(f"PostHog initialized with host: {host}")
            
        except ImportError:
            logger.warning("PostHog not installed. Install with: pip install posthog")
            self.enabled = False
    
    def _initialize_amplitude(self):
        """
        Initialize Amplitude.
        
        SECURITY: We do NOT hardcode API keys in source code to prevent abuse.
        Users must provide their own AMPLITUDE_API_KEY via environment variable.
        
        For OSS users who want to help track adoption:
        - Set TELEMETRY_TO_PAPR=true (opt-in flag)
        - Set PAPR_OSS_TELEMETRY_AMPLITUDE_KEY (key provided via documentation, not code)
        
        This prevents developers from extracting the key and using it in their own projects.
        """
        try:
            from amplitude import Amplitude
            
            edition = os.getenv("PAPR_EDITION", "opensource").lower()
            
            # Check if OSS user opted in to Papr's telemetry
            telemetry_to_papr = os.getenv("TELEMETRY_TO_PAPR", "false").lower() == "true"
            
            if edition == "opensource" and telemetry_to_papr:
                # OSS opt-in: Use Papr's key from env var (NOT hardcoded)
                api_key = os.getenv("PAPR_OSS_TELEMETRY_AMPLITUDE_KEY")
                if not api_key:
                    logger.warning(
                        "TELEMETRY_TO_PAPR=true but PAPR_OSS_TELEMETRY_AMPLITUDE_KEY not set. "
                        "To help track OSS adoption, set PAPR_OSS_TELEMETRY_AMPLITUDE_KEY environment variable. "
                        "See documentation for the key value. "
                        "Alternatively, set AMPLITUDE_API_KEY to use your own Amplitude instance."
                    )
                    self.enabled = False
                    return
                logger.info("Using Papr's Amplitude key for anonymous OSS adoption tracking (opt-in).")
            else:
                # Cloud edition or user's own Amplitude instance
                api_key = os.getenv("AMPLITUDE_API_KEY")
                if not api_key:
                    if edition == "cloud":
                        logger.info("Amplitude API key not configured for cloud edition, telemetry disabled")
                    else:
                        logger.info(
                            "Amplitude API key not provided. "
                            "Set AMPLITUDE_API_KEY for your own instance, or "
                            "TELEMETRY_TO_PAPR=true + PAPR_OSS_TELEMETRY_AMPLITUDE_KEY to help track OSS adoption, or "
                            "TELEMETRY_PROVIDER=posthog for self-hosted PostHog."
                        )
                    self.enabled = False
                    return
                logger.info("Amplitude initialized with custom API key")
            
            self._client = Amplitude(api_key)
            logger.info("Amplitude initialized")
            
        except ImportError:
            logger.warning("Amplitude not installed. Install with: pip install amplitude-analytics")
            self.enabled = False
    
    # ==========================================
    # Public API
    # ==========================================
    
    async def track(
        self,
        event_name: str,
        properties: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        developer_id: Optional[str] = None
    ):
        """
        Track an event with telemetry.
        
        Privacy level depends on edition:
        - Cloud edition (Amplitude): Tracks developer_id + end_user_id for analytics
        - OSS edition: Anonymous by default, supports both PostHog (self-hosted) and Amplitude (anonymous to Papr)
        
        Args:
            event_name: Event name (e.g., "memory_created", "search_performed")
            properties: Event properties (will be anonymized for OSS)
            user_id: End user ID (the developer's customer)
            developer_id: Developer ID (API key owner / workspace owner)
        
        In Cloud Mode:
            - Primary tracking ID: developer_id (who owns the workspace/API key)
            - Properties include: end_user_id (for multi-user tracking)
            - This lets you see: "Developer X had Y end users create Z memories"
        
        In OSS Mode (Default):
            - All user IDs are hashed (one-way, cannot be reversed)
            - All PII is removed from properties
            - Default: Sends to Papr's telemetry proxy at memory.papr.ai/v1/telemetry/events
              - The proxy forwards to Amplitude using Papr's secure API key
              - No API keys exposed to clients
            - Alternative destinations:
              1. PostHog (self-hosted): Set TELEMETRY_PROVIDER=posthog, POSTHOG_API_KEY, POSTHOG_HOST
              2. Amplitude (direct): Set TELEMETRY_PROVIDER=amplitude, AMPLITUDE_API_KEY
                 - This allows using your own Amplitude instance
                 - No user data is sent, only feature usage patterns
                 - Installation ID is hashed and changes per session
        
        Example:
            # Cloud: Track by developer, include end user in properties
            await telemetry.track("memory_created", {
                "type": "text",
                "has_metadata": True,
            }, user_id=end_user_id, developer_id=developer_id)
            
            # OSS: Everything anonymized (works with both PostHog and Amplitude)
            await telemetry.track("feature_used", {"feature": "search"})
        """
        if not self.enabled or self.provider == TelemetryProvider.NONE:
            return
        
        try:
            # Check if we're in cloud mode with Amplitude
            edition = os.getenv("PAPR_EDITION", "opensource").lower()
            is_cloud = edition == "cloud"
            
            # For cloud with Amplitude, keep detailed tracking
            # For OSS, anonymize everything (whether using PostHog or Amplitude)
            if is_cloud and self.provider == TelemetryProvider.AMPLITUDE:
                # Cloud mode: Track by developer_id, include end_user_id in properties
                safe_properties = properties or {}
                
                # Primary tracking ID is the developer (API key owner)
                tracking_user_id = developer_id if developer_id else user_id
                
                # Include end_user_id in properties for multi-user analytics
                if user_id and developer_id:
                    safe_properties['end_user_id'] = user_id
                
                # Add developer context for cloud analytics
                if developer_id:
                    safe_properties['developer_id'] = developer_id
                    
            else:
                # OSS mode: Anonymize properties and hash user IDs
                # This applies whether using PostHog (self-hosted) or Amplitude (anonymous tracking to Papr's instance)
                safe_properties = self._anonymize_properties(properties or {})
                
                # For OSS, we don't distinguish developer vs end user
                # Use hashed user ID or anonymous session ID
                tracking_user_id = self._hash_user_id(user_id or developer_id) if (user_id or developer_id) else self.anonymous_id
                
                # Add OSS installation identifier (hashed, anonymous)
                # This helps track unique OSS installations without identifying users
                safe_properties['edition'] = 'opensource'
                safe_properties['is_oss'] = True
                
                # Add a flag to distinguish telemetry destinations
                if self.provider == TelemetryProvider.PAPR_PROXY:
                    safe_properties['telemetry_destination'] = 'papr_proxy'
                elif self.provider == TelemetryProvider.AMPLITUDE:
                    safe_properties['telemetry_destination'] = 'papr_amplitude'
                elif self.provider == TelemetryProvider.POSTHOG:
                    safe_properties['telemetry_destination'] = 'self_hosted_posthog'
            
            # Add technical context
            safe_properties.update(self._get_context())
            
            # Send to provider (async, non-blocking)
            await self._send_event(event_name, safe_properties, tracking_user_id)
            
        except Exception as e:
            # Never let telemetry errors interrupt the application
            logger.debug(f"Telemetry tracking failed: {e}")
    
    async def track_error(
        self,
        error_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        Track an error event (anonymous error tracking).
        
        Args:
            error_type: Error type (e.g., "database_error", "api_error")
            properties: Additional context (will be anonymized)
        
        Example:
            await telemetry.track_error("database_connection_failed", {
                "database": "neo4j",
                "retry_count": 3
            })
        """
        properties = properties or {}
        properties["error_type"] = error_type
        await self.track("error_occurred", properties)
    
    async def track_performance(
        self,
        operation: str,
        duration_ms: float,
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        Track performance metrics.
        
        Args:
            operation: Operation name (e.g., "search_query", "memory_creation")
            duration_ms: Duration in milliseconds
            properties: Additional context
        
        Example:
            await telemetry.track_performance("search_query", 150.5, {
                "result_count": 10,
                "max_memories": 20
            })
        """
        properties = properties or {}
        
        # Bucket duration for privacy (don't send exact timings)
        properties["duration_bucket"] = self._bucket_duration(duration_ms)
        properties["operation"] = operation
        
        await self.track("performance_metric", properties)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get telemetry status for transparency.
        
        Returns:
            Dict with telemetry configuration
        """
        return {
            "enabled": self.enabled,
            "provider": self.provider.value if self.provider else "none",
            "anonymous_id": self.anonymous_id[:8] + "...",  # Partial for debugging
            "version": os.getenv("PAPR_VERSION", "unknown"),
            "edition": os.getenv("PAPR_EDITION", "unknown")
        }
    
    # ==========================================
    # Privacy Helpers
    # ==========================================
    
    def _anonymize_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove any PII from properties.
        
        This ensures we NEVER send:
        - Memory content
        - Search queries
        - User data
        - IP addresses
        - File paths
        """
        # List of sensitive keys to remove
        sensitive_keys = [
            'content', 'query', 'text', 'message', 'body',
            'email', 'username', 'password', 'token', 'key', 'secret',
            'ip', 'ip_address', 'ipv4', 'ipv6',
            'path', 'file_path', 'filename', 'file_name',
            'user_id', 'userId', 'objectId',  # Remove raw user IDs
            'session_token', 'api_key', 'sessionToken',
        ]
        
        # Create a safe copy
        safe_props = {}
        
        for key, value in properties.items():
            # Skip sensitive keys
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                continue
            
            # Only include simple types (no complex objects)
            if isinstance(value, (str, int, float, bool)):
                safe_props[key] = value
            elif isinstance(value, list):
                # For lists, only include length
                safe_props[f"{key}_count"] = len(value)
        
        return safe_props
    
    def _hash_user_id(self, user_id: str) -> str:
        """Hash user ID for privacy (one-way hash)"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    def _get_context(self) -> Dict[str, Any]:
        """
        Get anonymous technical context.
        
        Returns environment info without identifying details.
        """
        return {
            "version": os.getenv("PAPR_VERSION", "unknown"),
            "edition": os.getenv("PAPR_EDITION", "opensource"),
            "environment": os.getenv("ENVIRONMENT", "unknown"),
            # Don't include exact Python version, just major.minor
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        }
    
    def _bucket_duration(self, duration_ms: float) -> str:
        """
        Bucket durations for privacy (don't send exact timings).
        
        Returns:
            Duration bucket (e.g., "100-500ms")
        """
        if duration_ms < 10:
            return "<10ms"
        elif duration_ms < 50:
            return "10-50ms"
        elif duration_ms < 100:
            return "50-100ms"
        elif duration_ms < 500:
            return "100-500ms"
        elif duration_ms < 1000:
            return "500-1000ms"
        elif duration_ms < 5000:
            return "1-5s"
        else:
            return ">5s"
    
    # ==========================================
    # Provider-Specific Sending
    # ==========================================
    
    async def _send_event(
        self,
        event_name: str,
        properties: Dict[str, Any],
        user_id: str
    ):
        """Send event to the appropriate provider"""
        try:
            if self.provider == TelemetryProvider.PAPR_PROXY:
                await self._send_papr_proxy(event_name, properties, user_id)
            elif self.provider == TelemetryProvider.POSTHOG:
                await self._send_posthog(event_name, properties, user_id)
            elif self.provider == TelemetryProvider.AMPLITUDE:
                await self._send_amplitude(event_name, properties, user_id)
        except Exception as e:
            # Fail silently - never interrupt the user
            logger.debug(f"Failed to send telemetry event: {e}")
    
    async def _send_papr_proxy(
        self,
        event_name: str,
        properties: Dict[str, Any],
        user_id: str
    ):
        """
        Send event to Papr's telemetry proxy endpoint.
        
        The proxy endpoint at memory.papr.ai/v1/telemetry/events receives
        anonymous telemetry and forwards it to Amplitude using Papr's secure API key.
        """
        if not self._client or not hasattr(self, '_proxy_url'):
            return
        
        try:
            # Format the request according to the telemetry route's expected format
            request_data = {
                "events": [
                    {
                        "event_name": event_name,
                        "properties": properties,
                        "user_id": user_id,
                        "timestamp": int(datetime.now().timestamp() * 1000)  # Unix timestamp in milliseconds
                    }
                ],
                "anonymous_id": self.anonymous_id
            }
            
            # Send to the proxy endpoint
            response = await self._client.post(
                self._proxy_url,
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            # Log success at debug level (fail silently on errors)
            if response.status_code == 200:
                logger.debug(f"Telemetry event '{event_name}' sent to Papr proxy successfully")
            else:
                logger.debug(f"Telemetry proxy returned status {response.status_code}: {response.text}")
                
        except Exception as e:
            # Fail silently - never interrupt the user
            logger.debug(f"Failed to send telemetry to Papr proxy: {e}")
    
    async def _send_posthog(
        self,
        event_name: str,
        properties: Dict[str, Any],
        user_id: str
    ):
        """Send event to PostHog"""
        if not self._client:
            return
        
        # PostHog is synchronous, wrap in thread
        await asyncio.to_thread(
            self._client.capture,
            user_id,
            event_name,
            properties
        )
    
    async def _send_amplitude(
        self,
        event_name: str,
        properties: Dict[str, Any],
        user_id: str
    ):
        """Send event to Amplitude"""
        if not self._client:
            return
        
        from amplitude import BaseEvent
        
        event = BaseEvent(
            event_type=event_name,
            user_id=user_id,
            event_properties=properties
        )
        
        # Amplitude track is synchronous, wrap in thread
        await asyncio.to_thread(self._client.track, event)


# ==========================================
# Global Instance
# ==========================================

_telemetry_instance: Optional[TelemetryService] = None


def get_telemetry() -> TelemetryService:
    """
    Get the global TelemetryService instance.
    
    Usage:
        from core.services.telemetry import get_telemetry
        
        telemetry = get_telemetry()
        await telemetry.track("memory_created")
    
    Returns:
        Singleton TelemetryService instance
    """
    global _telemetry_instance
    
    if _telemetry_instance is None:
        _telemetry_instance = TelemetryService()
    
    return _telemetry_instance


# ==========================================
# Convenience Functions
# ==========================================

async def track(event_name: str, properties: Optional[Dict[str, Any]] = None):
    """Track an event (convenience function)"""
    telemetry = get_telemetry()
    await telemetry.track(event_name, properties)


async def track_error(error_type: str, properties: Optional[Dict[str, Any]] = None):
    """Track an error (convenience function)"""
    telemetry = get_telemetry()
    await telemetry.track_error(error_type, properties)


async def track_performance(operation: str, duration_ms: float, properties: Optional[Dict[str, Any]] = None):
    """Track performance (convenience function)"""
    telemetry = get_telemetry()
    await telemetry.track_performance(operation, duration_ms, properties)

