"""
Base Subscription Service Interface

This provides the interface for subscription management.
- Open source: No restrictions (unlimited access)
- Cloud: Real subscription checking via Stripe
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class SubscriptionService(ABC):
    """
    Base interface for subscription services.
    
    In open source edition, this provides unlimited access.
    In cloud edition, this is implemented by StripeSubscriptionService.
    """
    
    @abstractmethod
    async def check_user_subscription(self, user_id: str) -> Dict[str, Any]:
        """
        Check user's subscription status.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with subscription info:
            {
                "tier": str,           # Subscription tier
                "active": bool,        # Is subscription active
                "features": list,      # Available features
                "limits": dict        # Usage limits
            }
        """
        pass
    
    @abstractmethod
    async def check_interaction_limits(
        self,
        user_id: str,
        interaction_type: str = 'mini'
    ) -> Optional[tuple]:
        """
        Check if user has remaining interactions.
        
        Args:
            user_id: User identifier
            interaction_type: Type of interaction ('mini' or 'premium')
            
        Returns:
            None if within limits, or (error_dict, status_code, bool) if exceeded
        """
        pass
    
    async def get_user_tier(self, user_id: str) -> str:
        """Get user's subscription tier"""
        subscription = await self.check_user_subscription(user_id)
        return subscription.get("tier", "free")


class OpenSourceSubscriptionService(SubscriptionService):
    """
    Open source implementation - no restrictions.
    
    This is the default for self-hosted deployments.
    Everyone gets unlimited access.
    """
    
    async def check_user_subscription(self, user_id: str) -> Dict[str, Any]:
        """
        Open source: Everyone has unlimited access.
        """
        return {
            "tier": "unlimited",
            "active": True,
            "features": ["all"],
            "limits": {
                "max_memories": None,  # Unlimited
                "max_api_calls": None,  # Unlimited
                "rate_limit": None      # Unlimited
            }
        }
    
    async def check_interaction_limits(
        self,
        user_id: str,
        interaction_type: str = 'mini'
    ) -> Optional[tuple]:
        """
        Open source: No limits on interactions.
        """
        # No limits in open source
        return None
    
    async def get_user_tier(self, user_id: str) -> str:
        """Open source: Everyone is unlimited tier"""
        return "unlimited"


# Singleton instance
_subscription_service: Optional[SubscriptionService] = None


def get_subscription_service() -> SubscriptionService:
    """
    Get the configured subscription service.
    
    This will return either:
    - StripeSubscriptionService (cloud edition)
    - OpenSourceSubscriptionService (open source edition)
    
    Returns:
        Configured SubscriptionService instance
    """
    global _subscription_service
    
    if _subscription_service is None:
        # Try to load from app state first (set in app_factory)
        # Otherwise default to open source
        _subscription_service = OpenSourceSubscriptionService()
    
    return _subscription_service


def set_subscription_service(service: SubscriptionService):
    """
    Set the subscription service instance.
    
    Called by app_factory.py during app initialization.
    """
    global _subscription_service
    _subscription_service = service

