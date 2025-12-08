import stripe
import os  # Add this import
from os import environ as env
from services.logging_config import get_logger
from services.url_utils import clean_url
import asyncio
from datetime import datetime
from dotenv import find_dotenv, load_dotenv
from uuid import uuid4
import traceback

from services.logger_singleton import LoggerSingleton

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)

# Check if Stripe is enabled via feature flags
def _is_stripe_enabled():
    """Check if Stripe is enabled in the configuration"""
    try:
        from config import get_features
        features = get_features()
        return features.is_enabled("stripe")
    except Exception:
        # If config system not available, check env var as fallback
        return env.get("STRIPE_SECRET_KEY") is not None

class StripeService:
    _instance = None
    _client = None
    _api_key = None
    _secret_key = None
    
    # Map product IDs to tiers
    PRODUCT_TIER_MAP = {
        # Active products
        'prod_SVwgiBr4mps4tm': 'developer',
        'prod_SVUS311qKTCai7': 'starter',
        'prod_SVdHs09gaUV5Zb': 'growth',
        'prod_REdMj6jKqYsvg0': 'pro',
        'prod_RF7USgKdgJ7c61': 'business_plus',  # Business Plus
        'prod_RReMt8mN7hn4KT': 'enterprise',  # Enterprise
        # Note: Removed orphaned products that no longer exist in Stripe:
        # - prod_RIPUkyBY4dMZpX (old pro)
        # - prod_RIPVIBFgS4K7jh (old business_plus)
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StripeService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize Stripe configuration"""
        # Check if Stripe is enabled before initializing
        if not _is_stripe_enabled():
            logger.debug("Stripe is disabled in configuration - skipping initialization")
            self._client = None
            self._api_key = None
            self._secret_key = None
            return
        
        # Respect USE_DOTENV setting for consistency
        use_dotenv = env.get("USE_DOTENV", "true").lower() == "true"
        if use_dotenv:
            ENV_FILE = find_dotenv()
            if ENV_FILE:
                load_dotenv(ENV_FILE)

        # Initialize both API keys
        self._api_key = clean_url(env.get("STRIPE_API_KEY"))
        logger.debug(f"Initialized STRIPE_API_KEY: {self._api_key}")
        self._secret_key = clean_url(env.get('STRIPE_SECRET_KEY'))  # Secret key
        logger.debug(f"Initialized STRIPE_SECRET_KEY: {self._secret_key}")
        
        if not self._api_key or not self._secret_key:
            logger.debug("Missing Stripe API keys. Stripe features will be disabled.")
            self._client = None
            return
        
        # Initialize stripe with the secret key
        try:
            stripe.api_key = self._secret_key
            self._client = stripe.StripeClient(self._secret_key)
        except Exception as e:
            logger.warning(f"Failed to initialize Stripe client: {e}. Stripe features will be disabled.")
            self._client = None
            return
        
        # Log meter configuration (only if client is initialized)
        if self._client:
            try:
                # Get all meters configuration
                meters = self._client.billing.meters.list()
                logger.info("Available Stripe meters configuration:")
                for meter in meters:
                    logger.debug(f"""
                        Meter ID: {meter.id}
                        Name: {meter.display_name}
                        Customer Mapping Key: {meter.customer_mapping.event_payload_key if hasattr(meter, 'customer_mapping') else 'N/A'}
                        Value Key: {meter.value_settings.event_payload_key if hasattr(meter, 'value_settings') else 'N/A'}
                        Full config: {meter}
                    """)
            except Exception as e:
                logger.error(f"Error fetching meter configuration: {str(e)}")
            logger.info("Initialized Stripe service")
        else:
            logger.debug("Stripe service initialized but client is disabled")
        logger.debug(f"Initialized PRODUCT_TIER_MAP: {self.PRODUCT_TIER_MAP}")

    @property
    def client(self):
        """Get the Stripe client instance"""
        return self._client

    async def get_customer_tier(self, stripe_customer_id):
        """
        Get the customer's current subscription tier from Stripe
        Returns: 'pro', 'business_plus', 'enterprise', 'developer', 'starter', 'growth' or None for free trial
        """
        if not self._client:
            logger.debug("Stripe client not initialized - cannot get customer tier")
            return None
        
        if not self._secret_key:
            logger.debug("Missing Stripe secret key - cannot get customer tier")
            return None
        
        try:
            # Get subscription items using the proper Stripe API method
            # First get the subscription, then get its items separately
            subscriptions = await asyncio.to_thread(
                stripe.Subscription.list,
                customer=stripe_customer_id,
                status='active',
                limit=1
            )
            
            if not subscriptions or not subscriptions.data:
                logger.info(f"No active subscriptions found for customer {stripe_customer_id}")
                return None
            
            # Get the first active subscription
            subscription = subscriptions.data[0]
            logger.info(f"Found subscription: {subscription.id}")
            
            # Get subscription items using the SubscriptionItem API
            subscription_items = await asyncio.to_thread(
                stripe.SubscriptionItem.list,
                subscription=subscription.id
            )
            
            items_data = subscription_items.data if subscription_items else None
            logger.info(f"Retrieved {len(items_data) if items_data else 0} subscription items")
            
            if not items_data:
                logger.warning(f"No subscription items found for subscription {subscription.id}")
                return None
            
            # Get the first subscription item (contains the price)
            item = items_data[0]
            
            # Get the product ID from the price object
            if not item.price or not hasattr(item.price, 'product'):
                logger.warning(f"No product found in subscription item")
                return None
                
            product_id = item.price.product
            logger.info(f"Found product ID: {product_id}")
            
            # Use the existing product tier mapping
            tier = self.PRODUCT_TIER_MAP.get(product_id)
            if tier:
                logger.info(f"Successfully mapped customer {stripe_customer_id} to tier: {tier}")
                return tier
            else:
                logger.warning(f"Unknown product ID {product_id} for customer {stripe_customer_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting customer tier from Stripe: {e}")
            logger.error(f"Full traceback:")
            import traceback
            traceback.print_exc()
            return None

    async def send_meter_event(self, event_name: str, stripe_customer_id: str, value: int = 1):
        """
        Sends a meter event to Stripe using v1 API
        Returns: Response object if successful, None if failed
        """
        if not self._client:
            logger.debug("Stripe client not initialized - skipping meter event")
            return None
        
        if not self._secret_key:
            logger.debug("Missing Stripe secret key - skipping meter event")
            return None
        
        if not stripe_customer_id:
            logger.error("stripe_customer_id is missing.")
            return None

        try:
            # Configure the client with the secret key
            stripe.api_key = self._secret_key
            
            # Log the meter configuration for this specific event (background task to avoid blocking)
            async def log_meter_config():
                try:
                    if self._client:
                        meters = await asyncio.to_thread(self._client.billing.meters.list)
                    relevant_meter = next((m for m in meters if m.display_name == event_name), None)
                    if relevant_meter:
                        logger.info(f"Using meter '{event_name}' - customer_key: {relevant_meter.customer_mapping.event_payload_key if hasattr(relevant_meter, 'customer_mapping') else 'N/A'}")
                except Exception as e:
                    logger.debug(f"Meter config lookup failed: {str(e)}")  # Changed to debug to reduce noise

            # Run meter config logging in background (don't block on it)
            asyncio.create_task(log_meter_config())
            
            event_identifier = f"evt_{uuid4().hex}"
            
            event_data = {
                "event_name": event_name,
                "identifier": event_identifier,
                "payload": {
                    "stripe_customer_id": stripe_customer_id,
                    "value": str(value)
                }
            }
            
            logger.info(f"Sending meter event with data: {event_data}")
            
            # Run the synchronous Stripe operation in a thread pool
            response = await asyncio.to_thread(
                stripe.billing.MeterEvent.create,
                **event_data
            )
        
            logger.info(f"Successful meter event response: {response}")
            return response
            
        except stripe.error.AuthenticationError as e:
            logger.warning(f"Stripe authentication error (possibly in test mode): {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error sending Stripe meter event: {str(e)}")
            logger.exception("Full traceback:")
            return None

# Create a singleton instance
# This will initialize gracefully even if Stripe is disabled (client will be None)
stripe_service = StripeService()